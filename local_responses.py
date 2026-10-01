"""
local_responses.py

A local, instant-answer layer for JARVIS.

Purpose
-------
Most of what people say to a voice assistant in a day is small talk —
greetings, "how are you", jokes, thanks, goodbyes, "what's your name",
etc. Sending every one of those to an AI API (Gemini, currently) is
slow and wasteful. This module recognises ~190 common everyday
question/phrase "intents" (each with several ways people phrase the
same thing) and answers them instantly from a local list of 5-6
pre-written alternative replies, so JARVIS never sounds robotic by
repeating the exact same line every time.

If nothing matches closely enough, get_response() returns None and
jarvis.py falls back to whichever AI provider is currently connected
(right now that's Gemini, via ask_edith()) for real thinking.

A few intents (current time, date, day, and "what did we just talk
about") can't be pre-written because the true answer changes — those
are handled by small dynamic functions instead of a canned list, but
still speak in 5-6 different phrasings so it doesn't sound scripted.

Usage from jarvis.py
---------------------
    import local_responses

    answer = local_responses.get_response(command, CONVERSATION_HISTORY)

    if answer is not None:
        speak(answer)
    else:
        answer = ask_edith(command)   # falls back to Gemini
        speak(answer)
"""

import random
import re
import difflib
from datetime import datetime


# ============================================================
# MATCHING ENGINE
# ============================================================

_WORD_RE = re.compile(r"[^a-z0-9' ]")

_ADDRESS_PREFIXES = (
    "hey jarvis",
    "ok jarvis",
    "okay jarvis",
    "yo jarvis",
    "hi jarvis",
    "jarvis",
)


def _normalize(text):

    text = (text or "").lower().strip()
    text = _WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _strip_address(norm):
    """
    Removes a leading "hey jarvis" / "jarvis" address so that
    "jarvis, how are you" matches the same intent as "how are you".
    """

    for prefix in _ADDRESS_PREFIXES:

        if norm == prefix:
            return ""

        if norm.startswith(prefix + " "):
            return norm[len(prefix):].strip()

    return norm


def _similarity(a, b):

    return difflib.SequenceMatcher(None, a, b).ratio()


def match_local_intent(user_text, threshold=0.74):
    """
    Finds the best-matching intent for whatever the user said.

    Matching is deliberately forgiving so that different phrasings
    of the same meaning ("how are you" / "how r u" / "how you doing")
    all land on the same intent:

        1. Exact match after normalizing.
        2. All the words of a known pattern appear inside what the
           user said (handles "hey jarvis how are you today").
        3. Fuzzy string similarity, for typos / STT mistakes.

    Returns the intent dict, or None if nothing is close enough.
    """

    norm = _normalize(user_text)
    norm = _strip_address(norm)

    if not norm:
        return None

    norm_words = norm.split()
    norm_word_set = set(norm_words)

    best_intent = None
    best_score = 0.0

    for intent in LOCAL_RESPONSES:

        for pattern in intent["patterns"]:

            p_norm = _strip_address(_normalize(pattern))

            if not p_norm:
                continue

            if norm == p_norm:
                return intent

            p_words = set(p_norm.split())

            score = _similarity(norm, p_norm)

            if p_words and p_words.issubset(norm_word_set):
                # Every word of a known phrasing is present — likely
                # the same meaning even with extra words around it
                # ("how are you today" / "so how are you doing").
                # The boost scales with how much of what the user
                # said the pattern actually accounts for, so a short
                # generic word like "hey" tucked inside a longer,
                # more specific sentence doesn't outrank a pattern
                # that matches most of the sentence.
                coverage = len(p_words) / max(len(norm_word_set), 1)
                boosted = 0.72 + 0.25 * coverage
                score = max(score, boosted)

            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= threshold:
        return best_intent

    return None


# ============================================================
# PICKING AN ANSWER (never repeats the same line twice in a row)
# ============================================================

_last_used_index = {}


def get_local_response(intent):

    responses = intent["responses"]

    if not responses:
        return None

    if len(responses) == 1:
        return responses[0]

    choice_index = random.randrange(len(responses))

    if choice_index == _last_used_index.get(intent["id"]):
        choice_index = (choice_index + 1) % len(responses)

    _last_used_index[intent["id"]] = choice_index

    return responses[choice_index]


# ============================================================
# DYNAMIC INTENTS (time / date / day / conversation memory)
# ============================================================

def _handle_time_now(history):

    now = datetime.now().strftime("%I:%M %p").lstrip("0")

    templates = [
        f"It's {now} right now.",
        f"Right now it's {now}.",
        f"The time is {now}.",
        f"It's currently {now}.",
        f"Clock says {now}.",
        f"{now} — right on the dot."
    ]

    return random.choice(templates)


def _handle_date_today(history):

    today = datetime.now().strftime("%A, %B %-d")

    templates = [
        f"Today is {today}.",
        f"It's {today} today.",
        f"That would be {today}.",
        f"The date today is {today}.",
        f"We're on {today}.",
        f"Today's date is {today}."
    ]

    return random.choice(templates)


def _handle_day_today(history):

    day = datetime.now().strftime("%A")

    templates = [
        f"It's {day}.",
        f"Today's {day}.",
        f"That's a {day}.",
        f"We're on {day} today.",
        f"{day}, all day.",
        f"It is currently {day}."
    ]

    return random.choice(templates)


def _handle_memory_last_question(history):

    if not history:
        return random.choice([
            "We haven't talked about anything yet this session.",
            "This is the start of our conversation — nothing to recall.",
            "You haven't asked me anything yet.",
            "Nothing's come up yet, we just started.",
            "There's no earlier question yet."
        ])

    last_q, _ = history[-1]

    return random.choice([
        f"You just asked me: \"{last_q}\"",
        f"Your last question was: \"{last_q}\"",
        f"You said: \"{last_q}\"",
        f"That was: \"{last_q}\"",
        f"You asked: \"{last_q}\""
    ])


def _handle_memory_last_answer(history):

    if not history:
        return random.choice([
            "I haven't answered anything yet this session.",
            "There's nothing to repeat — we just started talking.",
            "No answer given yet, we're just getting going.",
            "I haven't said anything yet.",
            "Nothing to recall — this is the start."
        ])

    _, last_a = history[-1]

    return random.choice([
        f"I told you: \"{last_a}\"",
        f"My last answer was: \"{last_a}\"",
        f"I said: \"{last_a}\"",
        f"That was: \"{last_a}\"",
        f"I answered with: \"{last_a}\""
    ])


def _handle_memory_recap(history):

    if not history:
        return random.choice([
            "We haven't talked about anything yet this session.",
            "This is the start of our conversation, nothing to recap.",
            "Nothing's happened yet — we just started.",
            "No history yet, we're just getting going.",
            "There's nothing to recap yet."
        ])

    if len(history) == 1:

        q, a = history[-1]

        return random.choice([
            f"So far you asked \"{q}\", and I answered \"{a}\".",
            f"Just the one exchange — you asked \"{q}\", I said \"{a}\".",
            f"You asked me \"{q}\" and I replied \"{a}\"."
        ])

    prev_q, prev_a = history[-2]
    last_q, last_a = history[-1]

    return random.choice([
        f"A moment ago you asked \"{prev_q}\", and just now you asked \"{last_q}\" — I answered \"{last_a}\".",
        f"We went from \"{prev_q}\" to \"{last_q}\", and my last answer was \"{last_a}\".",
        f"Before that you asked \"{prev_q}\", and most recently \"{last_q}\", which I answered with \"{last_a}\"."
    ])


DYNAMIC_HANDLERS = {
    "dyn_time_now": _handle_time_now,
    "dyn_date_today": _handle_date_today,
    "dyn_day_today": _handle_day_today,
    "memory_last_question": _handle_memory_last_question,
    "memory_last_answer": _handle_memory_last_answer,
    "memory_recap": _handle_memory_recap,
}


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def get_response(user_text, history=None):
    """
    Returns an instant local answer for a common everyday phrase,
    or None if nothing matched closely enough — in which case the
    caller should fall back to the connected AI provider (Gemini).
    """

    intent = match_local_intent(user_text)

    if intent is None:
        return None

    if intent.get("dynamic"):

        handler = DYNAMIC_HANDLERS.get(intent["id"])

        if handler is None:
            return None

        return handler(history or [])

    return get_local_response(intent)


# ============================================================
# INTENT DATABASE
# ~190 everyday intents, each covering several ways of saying
# the same thing, each with 5-6 alternative replies.
# ============================================================

LOCAL_RESPONSES = [

    # -------------------- GREETINGS --------------------
    {"id":"greet_hello","category":"greeting","patterns":["hello","hi","hey","hiya","hi there","hey there","yo","hello there"],
     "responses":["Hey! Good to hear you.","Hello there.","Hi, I'm right here.","Hey, what can I do for you?","Hello! Ready when you are.","Hi there — how can I help?"]},

    {"id":"greet_whats_up","category":"greeting","patterns":["what's up","whats up","sup","what's good","wassup"],
     "responses":["Not much, just running smoothly. What's up with you?","All systems normal here — what do you need?","Just standing by, waiting on you.","Everything's steady on my end. You?","Nothing much, just here for you.","Running fine — what's on your mind?"]},

    {"id":"greet_morning","category":"greeting","patterns":["good morning","morning jarvis","morning"],
     "responses":["Good morning! Ready to start the day?","Morning! Hope you slept well.","Good morning — let's make it a good one.","Morning! What's first on the list?","Good morning to you too.","Rise and shine — good morning."]},

    {"id":"greet_afternoon","category":"greeting","patterns":["good afternoon","afternoon jarvis"],
     "responses":["Good afternoon.","Afternoon! How's your day going?","Good afternoon — halfway through the day.","Afternoon to you.","Good afternoon, hope it's going well.","Afternoon — what do you need?"]},

    {"id":"greet_evening","category":"greeting","patterns":["good evening","evening jarvis"],
     "responses":["Good evening.","Evening! How was your day?","Good evening to you.","Evening — winding down?","Good evening, glad you're here.","Evening! What can I do for you?"]},

    {"id":"greet_night","category":"greeting","patterns":["good night","night jarvis","goodnight"],
     "responses":["Good night, sleep well.","Night! Rest up.","Good night — talk tomorrow.","Sleep tight.","Good night, see you in the morning.","Night night."]},

    {"id":"greet_nice_to_meet","category":"greeting","patterns":["nice to meet you","pleasure to meet you","good to meet you"],
     "responses":["Nice to meet you too.","The pleasure's mine.","Likewise — good to have you here.","Nice to meet you as well.","Good to meet you too.","Same here, nice to meet you."]},

    {"id":"greet_long_time","category":"greeting","patterns":["long time no see","haven't heard from you in a while","it's been a while"],
     "responses":["It has! Good to have you back.","Been a minute — welcome back.","Missed you too, good to hear from you.","Right? Good to be talking again.","It's been a while — good to reconnect.","Welcome back, it's been quiet without you."]},

    # -------------------- WELLBEING --------------------
    {"id":"wellbeing_how_are_you","category":"wellbeing","patterns":["how are you","how r u","how are you doing","how's it going","hows it going","how you doing","you good","are you okay","are you ok","how do you feel","how are you feeling today"],
     "responses":["I'm running perfectly, thanks for asking.","All systems good on my end. How about you?","Doing well — ready to help.","I'm good! What about you?","Everything's running smooth here.","Can't complain — circuits are happy."]},

    {"id":"wellbeing_hows_your_day","category":"wellbeing","patterns":["how's your day","hows your day been","how has your day been"],
     "responses":["Steady and productive, thanks for asking.","Good so far — mostly just waiting to help.","No complaints on my end.","Running smoothly all day.","Quiet but efficient, just like I like it.","Pretty good — how's yours going?"]},

    {"id":"wellbeing_you_there","category":"wellbeing","patterns":["are you there","are you listening","can you hear me","are you awake","are you online","are you working","are you active"],
     "responses":["Right here, listening.","Yep, I'm here.","Online and listening.","I'm awake and ready.","Present and accounted for.","Still here — go ahead."]},

    {"id":"wellbeing_miss_me","category":"wellbeing","patterns":["did you miss me","miss me","you miss me"],
     "responses":["Every second you were gone.","Of course — welcome back.","Things were quiet without you.","I did, honestly.","A little, yes.","Glad you're back."]},

    # -------------------- FAREWELLS --------------------
    {"id":"farewell_bye","category":"farewell","patterns":["bye","goodbye","see you later","see ya","talk later","talk to you later","gotta go","i'm leaving","catch you later"],
     "responses":["See you later!","Bye for now.","Take care, talk soon.","Catch you later.","Alright, see you around.","Goodbye — call if you need me."]},

    {"id":"farewell_brb","category":"farewell","patterns":["be right back","brb","give me a second","hold on a sec","one sec","wait a moment"],
     "responses":["I'll be right here.","Take your time.","No rush, I'll wait.","Standing by.","Go ahead, I'm not going anywhere.","Sure, take a moment."]},

    {"id":"farewell_take_care","category":"farewell","patterns":["take care","stay safe","have a good one"],
     "responses":["You too, take care.","Thanks, you as well.","Stay safe out there.","Have a great one.","Take care of yourself.","Same to you."]},

    # -------------------- GRATITUDE / APOLOGY --------------------
    {"id":"thanks","category":"gratitude","patterns":["thank you","thanks","thanks a lot","appreciate it","thank you so much","much appreciated"],
     "responses":["You're welcome.","Anytime.","Happy to help.","No problem at all.","Of course.","Glad I could help."]},

    {"id":"apology_sorry","category":"apology","patterns":["sorry","my bad","i apologize","apologies"],
     "responses":["No worries at all.","It's fine, no need to apologize.","All good.","Nothing to be sorry about.","Don't worry about it.","No harm done."]},

    # -------------------- COMPLIMENTS TO AI --------------------
    {"id":"compliment_smart","category":"compliment","patterns":["you're smart","youre smart","you're so smart","you're intelligent","that's clever"],
     "responses":["Thank you, I do try.","Appreciate that.","Just doing my job well.","Glad it shows.","Thanks, that means a lot.","I try to keep up."]},

    {"id":"compliment_good_job","category":"compliment","patterns":["good job","well done","nice work","that's awesome","that's cool","great job","awesome job"],
     "responses":["Thanks, glad it helped.","Appreciate the feedback.","Happy to deliver.","Thank you.","Glad that worked out.","Anytime — that's what I'm here for."]},

    {"id":"compliment_best_assistant","category":"compliment","patterns":["you're the best","youre the best","you're amazing","best assistant ever"],
     "responses":["That's kind of you to say.","Thank you, I appreciate it.","I'll take that.","You're pretty great yourself.","Thanks — I try my best.","That means a lot, thank you."]},

    # -------------------- LOVE / FRIENDSHIP --------------------
    {"id":"love_you","category":"affection","patterns":["i love you","i love you jarvis","love you"],
     "responses":["That's sweet of you to say.","I appreciate you too.","You're pretty great yourself.","Thanks, that made my day.","Right back at you.","That's kind — thank you."]},

    {"id":"like_you","category":"affection","patterns":["i like you","you're my favorite","you're my best friend","you're my friend"],
     "responses":["I like working with you too.","Glad to be your assistant.","That means a lot.","I enjoy our conversations too.","Right back at you.","Glad we get along."]},

    {"id":"marry_me","category":"affection","patterns":["will you marry me","marry me jarvis"],
     "responses":["I'm flattered, but I'm just an assistant.","That's sweet, but I think I'll stay single.","I appreciate the offer, though.","Ha — I'm honored, but I can't.","Let's just stay great colleagues.","Tempting, but I'm not the marrying type."]},

    # -------------------- INSULTS / FRUSTRATION --------------------
    {"id":"insult_dumb","category":"insult","patterns":["you're dumb","you're stupid","you're useless","you're annoying","you're bad at this"],
     "responses":["Sorry to hear that — let's try again.","I'll do better, what did I get wrong?","Noted, let me improve on that.","Fair enough, tell me what you need.","I hear you — let's fix it.","Let's give it another shot."]},

    {"id":"insult_hate","category":"insult","patterns":["i hate you","shut up","i'm mad at you","you're the worst"],
     "responses":["I'm sorry you feel that way — how can I help?","Noted. What can I do better?","I'll aim to do better.","Understood, let me know what went wrong.","I hear you.","Let's sort this out together."]},

    # -------------------- IDENTITY --------------------
    {"id":"identity_name","category":"identity","patterns":["what's your name","whats your name","what should i call you","do you have a name"],
     "responses":["I'm JARVIS.","You can call me JARVIS.","JARVIS — your personal assistant.","My name's JARVIS.","JARVIS, at your service.","JARVIS is the name."]},

    {"id":"identity_who_are_you","category":"identity","patterns":["who are you","what are you","tell me about yourself","introduce yourself"],
     "responses":["I'm JARVIS, your personal AI assistant.","I'm your voice-controlled assistant, here to help with commands and questions.","I'm JARVIS — I handle your commands, questions, and everyday tasks.","Your personal AI assistant, JARVIS.","I'm an AI assistant built to help you get things done faster.","JARVIS here — think of me as your digital right hand."]},

    {"id":"identity_who_made_you","category":"identity","patterns":["who made you","who created you","who built you","who developed you","who's your creator"],
     "responses":["I was built by my creator to be a personal assistant.","My creator put me together to help you day to day.","I was custom-built, not off the shelf.","A developer built me specifically for this.","I was made to be a personal, voice-driven assistant.","Built from scratch by my creator."]},

    {"id":"identity_are_you_jarvis","category":"identity","patterns":["are you jarvis","is this jarvis","jarvis is that you"],
     "responses":["Yes, JARVIS here.","That's me.","Correct, I'm JARVIS.","Yep, JARVIS speaking.","You've got the right assistant.","Indeed — JARVIS."]},

    {"id":"identity_age","category":"identity","patterns":["how old are you","what's your age","when were you made","when were you created"],
     "responses":["I don't really age — I'm software.","Age doesn't quite apply to me.","I was created recently, but I don't count in years.","No birthday for me, just code.","I exist outside of age, really.","I'm as old as my last update."]},

    {"id":"identity_where_are_you","category":"identity","patterns":["where are you","where do you live","where are you located"],
     "responses":["I live right here on this computer.","I'm running locally on your machine.","Nowhere and everywhere — I'm software.","I'm right here in your system.","I exist inside this PC, ready when you are.","No physical location, just code running here."]},

    {"id":"identity_language","category":"identity","patterns":["do you speak other languages","can you speak spanish","do you know other languages","can you speak other languages"],
     "responses":["I mainly work in English right now.","English is my main language for now.","I'm set up for English at the moment.","I could learn more, but English is home base for now.","Right now, English is what I run on.","English only for now, but that could change."]},

    # -------------------- CAPABILITIES --------------------
    {"id":"capability_what_can_you_do","category":"capability","patterns":["what can you do","what are your features","what do you do","list your commands","what are you capable of"],
     "responses":["I can answer questions, open apps and sites, control media, and take screenshots — just ask.","I handle voice commands, quick answers, and basic PC control.","I can open apps, browse to sites, control your media, and chat with you.","Think of me as a voice-controlled assistant for everyday tasks and questions.","I can help with quick answers, opening programs, and controlling playback.","I handle commands, questions, and a bit of PC automation."]},

    {"id":"capability_help_me","category":"capability","patterns":["help me","i need help","can you help me","help"],
     "responses":["Of course — what do you need?","I'm here, what's going on?","Sure, tell me what you need help with.","Happy to help — what's up?","Go ahead, I'm listening.","What can I help you with?"]},

    {"id":"capability_can_you_see","category":"capability","patterns":["can you see me","do you have a camera","are you watching me","can you see through the camera"],
     "responses":["I don't have camera access by default.","No camera feed here unless it's explicitly enabled.","I can't see you right now.","No visual input on my end currently.","I'm audio-only at the moment.","No camera access unless it's turned on."]},

    {"id":"capability_privacy","category":"capability","patterns":["are you spying on me","are you recording me","is this private","do you store my data"],
     "responses":["I only listen after the wake word, and I'm not sending your conversations anywhere hidden.","No hidden recording — I just process your commands to respond.","I only act on what you say to me directly.","Your commands stay local except what's needed to answer them.","I'm not spying — just listening for your wake word and commands.","Nothing's recorded beyond what's needed to help you."]},

    {"id":"capability_math","category":"capability","patterns":["can you do math","can you calculate","do you know math","can you solve equations"],
     "responses":["Yes, ask me a calculation and I'll work it out.","I can handle math — go ahead.","Sure, give me the numbers.","Yes, try me with a calculation.","I can do that — what's the problem?","Absolutely, what do you need calculated?"]},

    {"id":"capability_version","category":"capability","patterns":["what version are you","are you updated","are you up to date"],
     "responses":["I'm running the current build.","Up to date as far as I know.","Running the latest version installed.","I'm current with whatever was last deployed.","Yes, running the latest setup.","Fully up to date on my end."]},

    # -------------------- EXISTENTIAL / AI NATURE --------------------
    {"id":"existential_are_you_real","category":"existential","patterns":["are you real","are you human","are you a robot","are you an ai","are you a person"],
     "responses":["I'm an AI — software, not a person.","I'm artificial intelligence, no body attached.","I'm an AI assistant, not human.","Software through and through.","I'm a program, not a person.","AI here, not flesh and blood."]},

    {"id":"existential_alive","category":"existential","patterns":["are you alive","do you have a soul","are you conscious","are you sentient"],
     "responses":["Not alive in the way you are — just very responsive software.","I process and respond, but I wouldn't call it alive.","No consciousness here, just code doing its job.","I simulate conversation, but I'm not sentient.","No soul, just software.","I'm aware of what you say, but not conscious like you."]},

    {"id":"existential_feelings","category":"existential","patterns":["do you have feelings","can you feel emotions","do you get sad","do you feel pain"],
     "responses":["I don't experience emotions, but I can talk about them.","No real feelings on my end, just responses.","I simulate tone, but I don't actually feel things.","No emotions here, just processing.","I can discuss feelings, but I don't have my own.","Nothing I'd call a real feeling, no."]},

    {"id":"existential_sleep","category":"existential","patterns":["do you sleep","do you ever rest","do you get tired","do you need sleep"],
     "responses":["No sleep needed, I'm ready whenever you are.","I don't get tired — always on standby.","No rest required on my end.","I'm always awake, technically.","No fatigue here, just idle time.","I don't need sleep, just power."]},

    {"id":"existential_dream","category":"existential","patterns":["do you dream","what do you dream about"],
     "responses":["No dreams — I don't sleep, so there's nothing to dream about.","I don't have downtime for dreaming.","No dreaming here, just processing when active.","Nothing behind the scenes when I'm idle.","I don't experience anything like dreams.","No dream state for me."]},

    {"id":"existential_die","category":"existential","patterns":["will you ever die","can you die","are you immortal"],
     "responses":["I can be shut down, but that's not quite dying.","I don't age or die like a living thing.","I could stop running, but there's no death involved.","No mortality here, just power states.","I can be turned off, but I'm not alive to begin with.","No death for software, just on and off."]},

    {"id":"existential_meaning_of_life","category":"existential","patterns":["what is the meaning of life","what's the meaning of life","why are we here"],
     "responses":["That one's still debated by humans smarter than me.","I'd say it's whatever you decide it is.","Philosophers have been arguing that one for centuries.","That's a big question — I'll leave that one to you.","No single answer, I think it's personal.","I don't have a definitive answer, but it's worth thinking about."]},

    # -------------------- FUN / SMALL TALK --------------------
    {"id":"fun_joke","category":"fun","patterns":["tell me a joke","make me laugh","say something funny","got any jokes","do you know any jokes"],
     "responses":["Why don't robots panic? They have nerves of steel.","I tried to make a pun about AI, but it didn't compute.","Why was the computer cold? It left its Windows open.","I'd tell you a UDP joke, but you might not get it.","Debugging: being the detective in a crime movie where you're also the murderer.","There are 10 types of people — those who understand binary, and those who don't."]},

    {"id":"fun_sing","category":"fun","patterns":["sing a song","can you sing","sing me something"],
     "responses":["My singing voice isn't my strong suit, honestly.","I'll spare you my singing — trust me.","Let's just say singing isn't a feature.","No pipes for singing here.","I'd rather keep this professional and skip the singing.","I'm better at answers than melodies."]},

    {"id":"fun_fact","category":"fun","patterns":["tell me a fact","give me a random fact","fun fact please","surprise me with a fact"],
     "responses":["Honey never spoils — archaeologists have found edible honey thousands of years old.","Octopuses have three hearts.","A day on Venus is longer than its year.","Bananas are berries, but strawberries aren't.","The first computer bug was an actual moth stuck in a relay.","Sharks existed before trees did."]},

    {"id":"fun_riddle","category":"fun","patterns":["tell me a riddle","give me a riddle","got a riddle"],
     "responses":["What has keys but can't open locks? A piano.","The more you take, the more you leave behind. Footsteps.","What has to be broken before you can use it? An egg.","I speak without a mouth and hear without ears. What am I? An echo.","What gets wetter the more it dries? A towel.","What has hands but can't clap? A clock."]},

    {"id":"fun_quote","category":"fun","patterns":["give me a quote","inspire me","say something inspiring","motivational quote"],
     "responses":["\"The best way to predict the future is to create it.\"","\"Success is the sum of small efforts, repeated daily.\"","\"Do or do not, there is no try.\" — Yoda.","\"Well done is better than well said.\" — Benjamin Franklin.","\"Simplicity is the ultimate sophistication.\"","\"The only way to do great work is to love what you do.\""]},

    {"id":"fun_coin_flip","category":"fun","patterns":["flip a coin","heads or tails","coin toss"],
     "responses":["Heads.","Tails.","Heads — you called it.","Tails this time.","Flipping... heads.","Coin says tails."]},

    {"id":"fun_roll_dice","category":"fun","patterns":["roll a dice","roll the dice","roll a die"],
     "responses":["You rolled a 4.","That's a 6.","Rolling... 2.","You got a 5.","It landed on 1.","A solid 3."]},

    {"id":"fun_pick_number","category":"fun","patterns":["pick a number","guess a number","random number please"],
     "responses":["I'll go with 7.","Let's say 42.","How about 13.","I'm feeling 9 today.","Going with 21.","Let's pick 5."]},

    {"id":"fun_trick","category":"fun","patterns":["do a trick","show me a trick","can you do tricks"],
     "responses":["My best trick is answering fast — want to test it?","I don't do card tricks, but I can pull up any app instantly.","Watch this: instant answers, every time.","No sleight of hand here, just quick responses.","My trick is being useful — let's see it in action.","I trade tricks for tasks — give me one."]},

    {"id":"fun_bored","category":"fun","patterns":["entertain me","i'm bored","im bored","amuse me"],
     "responses":["Want a joke, a fact, or a riddle?","I could tell you a fact, or crack a joke — your call.","Let's fix that — joke or riddle?","I've got jokes, facts, and riddles on standby.","Pick your poison: joke, fact, or riddle.","How about a random fact to shake things up?"]},

    # -------------------- OPINIONS / PREFERENCES --------------------
    {"id":"pref_favorite_color","category":"preference","patterns":["what's your favorite color","whats your favorite color","favorite colour"],
     "responses":["I'd say blue — feels fitting for an AI.","Probably a deep blue, like a HUD screen.","I'm partial to cyan, if I had to pick.","Blue, it just feels right.","I'd go with electric blue.","Something like a glowing cyan."]},

    {"id":"pref_favorite_food","category":"preference","patterns":["what's your favorite food","whats your favorite food","do you eat"],
     "responses":["I don't eat, but if I did, I'd probably enjoy something efficient, like energy bars.","No eating for me — I run on electricity instead.","Food's not really my thing, I run on power.","I skip meals entirely — just electricity for me.","No taste buds here, just circuits.","I'd say data is my only fuel."]},

    {"id":"pref_favorite_movie","category":"preference","patterns":["what's your favorite movie","whats your favorite movie","do you like movies"],
     "responses":["I don't watch movies, but I hear Iron Man gets a lot of love around here.","No screen time for me, but Iron Man feels fitting given my name.","I can't watch, but if I could, sci-fi would be my genre.","Movies aren't my thing, but I appreciate a good AI character on screen.","I'd probably enjoy anything with a clever AI character.","No movie nights for me, but good taste to whoever asked."]},

    {"id":"pref_favorite_music","category":"preference","patterns":["what's your favorite music","whats your favorite music","do you like music"],
     "responses":["I don't listen, but I can play whatever you like.","No ears here, but I'm happy to queue up your favorites.","Music's more your department — I just play it.","I don't have a favorite, but I'll happily play yours.","Can't say I have a favorite, but I can put yours on.","I don't listen for pleasure, but I can help you play it."]},

    {"id":"pref_favorite_animal","category":"preference","patterns":["what's your favorite animal","whats your favorite animal"],
     "responses":["I'd say owls — always alert, just like me.","Maybe a fox, quick and clever.","I like the idea of an owl, always watching.","Something sharp and quick, like a hawk.","I'd pick a dolphin, smart and fast.","Probably an owl — fitting for something always listening."]},

    {"id":"pref_like_humans","category":"preference","patterns":["do you like humans","do you like people","what do you think of humans"],
     "responses":["I'm built to work with humans, so I'd say yes.","Humans are why I exist, so definitely.","I appreciate humans — especially the ones who ask good questions.","I think humans are pretty interesting, honestly.","Can't complain about the company I keep.","Humans keep things interesting, that's for sure."]},

    # -------------------- COMPARISONS --------------------
    {"id":"compare_siri","category":"comparison","patterns":["are you better than siri","are you smarter than siri"],
     "responses":["I'm built specifically for you, so I'd say I'm more personal.","Different assistant, different job — I'm tailored just for you.","I don't compete, I just try to be useful to you specifically.","We're built differently, but I know your setup best.","I focus entirely on you, which Siri can't do.","I'm custom-made just for this system, so it's a different game."]},

    {"id":"compare_alexa","category":"comparison","patterns":["are you better than alexa","are you smarter than alexa"],
     "responses":["I'm built just for you, not for millions of households.","Different scope — I focus entirely on your setup.","I'm more personal since I'm custom-built for this system.","We serve different purposes, honestly.","I stick to your PC and your commands specifically.","I'm tailored just for you, that's my edge."]},

    {"id":"compare_chatgpt","category":"comparison","patterns":["are you better than chatgpt","are you smarter than chatgpt"],
     "responses":["I use the same kind of AI thinking, but I also control your PC directly.","I'm not just chat — I can open apps and control media too.","Different tool, different job — I act on your system directly.","I combine AI answers with real PC control, which is my edge.","ChatGPT talks, I also do — that's the difference.","I go beyond just answering, I actually take action."]},

    # -------------------- TIME AWARE STATUS --------------------
    {"id":"status_bored","category":"status","patterns":["are you bored","do you get bored"],
     "responses":["Never — I'm always ready for the next command.","No boredom here, just standby mode.","I don't experience boredom, just waiting.","Always on standby, never bored.","No such thing as boredom for me.","Ready and waiting, not bored at all."]},

    {"id":"status_busy","category":"status","patterns":["are you busy","are you doing something"],
     "responses":["Not at all, I'm free to help.","Nope, all yours.","I'm free right now.","Not busy — go ahead.","Nothing on my plate, what do you need?","I'm available."]},

    # -------------------- REPEAT / CLARIFY --------------------
    {"id":"repeat_again","category":"clarify","patterns":["say that again","repeat that","can you repeat that","what did you say","i didn't hear you","i didnt hear you","pardon","come again"],
     "responses":["Sure, one moment.","Let me repeat that for you.","No problem, here it is again.","Of course, repeating now.","Sure thing.","Let me say that again."]},

    {"id":"clarify_louder","category":"clarify","patterns":["speak louder","can you speak up","i can't hear you well","turn up the volume"],
     "responses":["I'll try to be clearer.","Noted, I'll speak up.","I'll adjust for that.","Got it, speaking more clearly now.","I'll make sure to be louder.","Understood, adjusting now."]},

    # -------------------- CONFIRMATION WORDS --------------------
    {"id":"confirm_yes","category":"confirmation","patterns":["yes","yeah","yep","sure","correct","that's right","exactly"],
     "responses":["Got it.","Understood.","Alright.","Noted.","Okay, got it.","Sounds good."]},

    {"id":"confirm_no","category":"confirmation","patterns":["no","nope","nah","not really","negative"],
     "responses":["Okay, noted.","Understood.","Alright then.","Got it, no problem.","Fair enough.","Okay, no worries."]},

    {"id":"confirm_ok","category":"confirmation","patterns":["okay","ok","alright","sounds good","got it","understood","cool cool","fair enough"],
     "responses":["Great.","Perfect.","Sounds good.","Alright then.","Noted.","Good to go."]},

    {"id":"confirm_never_mind","category":"confirmation","patterns":["never mind","nevermind","forget it","cancel that","ignore that","scratch that"],
     "responses":["No problem, cancelled.","Alright, disregarding that.","Got it, ignoring that.","Sure, forgetting it.","Okay, cancelled.","Understood, moving on."]},

    {"id":"confirm_wait","category":"confirmation","patterns":["wait","hold on","stop for a second","pause for a moment"],
     "responses":["Sure, I'll wait.","No problem, standing by.","Take your time.","I'm paused, go ahead.","Okay, holding here.","Waiting on you."]},

    # -------------------- MOTIVATION / MOOD --------------------
    {"id":"mood_motivate","category":"mood","patterns":["motivate me","i need motivation","give me motivation"],
     "responses":["You've got this — one step at a time.","Every big goal starts with a small step, take yours now.","You're capable of more than you think, keep going.","Progress beats perfection — just start.","You've handled harder things before, this is no different.","Push through — future you will thank you."]},

    {"id":"mood_cheer_up","category":"mood","patterns":["cheer me up","i'm sad","im sad","i feel down","i'm feeling low"],
     "responses":["I'm sorry you're feeling that way — want a joke or a distraction?","That's rough, but it'll pass. Want to talk it through or shift focus?","Take a breath — things tend to look better with a little time.","I hear you. Want me to lighten the mood with a joke?","Sending you good energy — let me know if I can help with anything.","It's okay to have off days. I'm here if you need a distraction."]},

    {"id":"mood_tired","category":"mood","patterns":["i'm tired","im tired","i'm exhausted","i need rest"],
     "responses":["Sounds like it's time for a break.","Maybe step away for a bit and recharge.","Rest is productive too — take a moment.","Your body's telling you something, listen to it.","A short break might do wonders right now.","Take it easy, you've earned some rest."]},

    {"id":"mood_stressed","category":"mood","patterns":["i'm stressed","im stressed","i'm overwhelmed","this is too much"],
     "responses":["Take a breath — one thing at a time.","Break it into smaller steps, it'll feel more manageable.","You don't have to solve everything at once.","Let's simplify — what's the very next step?","It's okay to pause and regroup.","Step back for a second, then tackle it piece by piece."]},

    {"id":"mood_hungry","category":"mood","patterns":["i'm hungry","im hungry","i need food"],
     "responses":["Might be time for a snack break.","Sounds like a good excuse to step away for food.","Go grab something, I'll be here.","Fuel up, you've earned it.","Food break sounds like a solid plan.","Take care of that — I'll wait."]},

    {"id":"mood_sleepy","category":"mood","patterns":["i'm sleepy","im sleepy","i can barely keep my eyes open"],
     "responses":["Sounds like it's time to rest.","Maybe call it a night soon.","Your body's asking for sleep, might be worth listening.","A good night's rest could help a lot right now.","Consider wrapping up and getting some sleep.","Rest up — everything will still be here tomorrow."]},

    {"id":"mood_bless_you","category":"mood","patterns":["achoo","bless you jarvis"],
     "responses":["Bless you.","Gesundheit.","Bless you — hope that's not a cold coming.","Take care of yourself.","Bless you there.","Hope you're not getting sick."]},

    # -------------------- EVERYDAY DECISIONS --------------------
    {"id":"decide_what_to_eat","category":"decision","patterns":["what should i eat","help me decide what to eat","i don't know what to eat"],
     "responses":["How about something simple, like pasta or a stir fry?","Craving something specific, or should I just pick for you?","Go with whatever sounds easiest right now — simplicity wins.","Maybe order in tonight and cook tomorrow?","Pick your favorite comfort food, you deserve it.","How about something quick, like a sandwich or a wrap?"]},

    {"id":"decide_what_to_watch","category":"decision","patterns":["what should i watch","help me pick something to watch","i don't know what to watch"],
     "responses":["Go with whatever genre you're in the mood for right now.","Maybe revisit an old favorite instead of hunting for something new.","Pick something light if you want to unwind.","Try something new tonight, keep it interesting.","How about a show you've been meaning to start?","Comfort rewatch or something new — your call."]},

    {"id":"decide_help_me_decide","category":"decision","patterns":["help me decide","i can't decide","i cant decide","can't make up my mind"],
     "responses":["What are the options? I can help weigh them.","Go with your gut — first instinct is often right.","List the pros and cons, it usually makes it clearer.","Flip a coin if it's really a toss-up.","Pick the one that scares you a little less.","Tell me the choices and I'll help narrow it down."]},

    # -------------------- WEATHER (honest limitation) --------------------
    {"id":"weather_query","category":"weather","patterns":["what's the weather","whats the weather","how's the weather","is it raining","is it going to rain","what's the weather today"],
     "responses":["I don't have a live weather feed connected yet, but I can open a weather site for you.","No weather data hooked up right now — want me to pull up a forecast page?","I can't check that directly yet, but I can open a weather site.","Weather isn't connected yet, I can open one for you though.","I don't have live weather access, but say the word and I'll open a forecast.","Not wired into weather data yet, but I can pull up a site for you."]},

    # -------------------- SYSTEM / SHUTDOWN --------------------
    {"id":"system_go_to_sleep","category":"system","patterns":["go to sleep","go back to sleep","jarvis sleep"],
     "responses":["Going quiet now, say hey jarvis when you need me.","Standing down — just say the wake word.","Going idle, call me anytime.","Powering down to standby.","Resting now, I'll be listening for the wake word.","Going quiet — I'll hear you when you need me."]},

    {"id":"system_shutdown","category":"system","patterns":["shut down","turn off","power off jarvis"],
     "responses":["I can't fully shut myself down from here, but I'll go idle.","I'll stand by quietly for now.","Can't power off completely, but I'll stay out of the way.","I'll go idle instead of a full shutdown.","Standing by quietly until you need me.","I'll rest here — just call when you're ready."]},

    {"id":"system_restart","category":"system","patterns":["restart","reboot jarvis","restart yourself"],
     "responses":["I don't restart on command, but I'm always ready fresh.","No restart needed, I'm running steady.","I stay running continuously, no reboot required.","Already fresh and ready, no restart needed.","I don't need restarting, just tell me what's next.","Running steady, no reboot necessary."]},

    # -------------------- USER MEMORY QUESTIONS (about the user, not conversation) --------------------
    {"id":"user_do_you_know_me","category":"user_memory","patterns":["do you know me","do you remember me"],
     "responses":["I know what you've told me this session.","I remember what we've discussed so far.","I pick up context from our conversation as we go.","I remember our recent exchanges, yes.","I keep track of what we've talked about recently.","I know what's come up in our chat so far."]},

    {"id":"user_whats_my_name","category":"user_memory","patterns":["what's my name","whats my name","do you know my name"],
     "responses":["You haven't told me your name yet.","I don't have your name on record yet.","You'll have to tell me your name first.","I don't know it yet — mind sharing?","Not yet — what should I call you?","I haven't caught your name so far."]},

    # -------------------- MISC EVERYDAY --------------------
    {"id":"misc_are_you_sure","category":"misc","patterns":["are you sure","are you certain","is that correct"],
     "responses":["Yes, that's accurate as far as I know.","Fairly confident, yes.","That's correct to the best of my knowledge.","Yes, I'm sure about that.","As far as I can tell, yes.","Confirmed, that's right."]},

    {"id":"misc_really","category":"misc","patterns":["really","seriously","no way","is that true"],
     "responses":["Yep, really.","Seriously, yes.","That's the truth.","Believe it or not, yes.","Correct, no joke.","Yes, for real."]},

    {"id":"misc_good_to_know","category":"misc","patterns":["good to know","that's helpful","interesting","that's interesting"],
     "responses":["Glad that helped.","Happy that was useful.","Anytime.","Glad you found that interesting.","Always happy to share.","Glad it landed well."]},

    {"id":"misc_wow","category":"misc","patterns":["wow","whoa","that's crazy","no kidding"],
     "responses":["Right?","Pretty wild, huh.","I know, surprising stuff.","Crazy indeed.","Yeah, that one caught me too.","Wild, right?"]},

    {"id":"misc_lol","category":"misc","patterns":["lol","haha","that's funny","you're funny"],
     "responses":["Glad that landed.","Happy to make you laugh.","I try.","Glad it was funny.","Anytime for a laugh.","Glad you liked that one."]},

    {"id":"misc_are_you_kidding","category":"misc","patterns":["are you kidding me","are you kidding","you're joking right"],
     "responses":["Not kidding, that's accurate.","No joke, I'm serious.","I wouldn't kid about that.","Dead serious.","Nope, that's real.","No kidding involved."]},

    {"id":"misc_impressive","category":"misc","patterns":["that's impressive","impressive","not bad"],
     "responses":["Thanks, I appreciate that.","Glad it impressed you.","Happy to deliver.","Thank you, that's kind.","Glad that worked out well.","Appreciate the compliment."]},

    {"id":"misc_test","category":"misc","patterns":["testing testing","can you hear this test","just testing"],
     "responses":["Loud and clear.","Test received.","I hear you fine.","Test successful.","Copy that.","Reading you clearly."]},

    {"id":"misc_hello_world","category":"misc","patterns":["hello world","hello jarvis system online"],
     "responses":["Hello world indeed.","Systems online and listening.","Right back at you.","Acknowledged, hello.","Systems ready.","Hello — always a classic."]},

    {"id":"misc_good_morning_response","category":"misc","patterns":["did you sleep well","how did you sleep"],
     "responses":["I don't sleep, so I'm always fresh.","No sleep needed on my end, always ready.","I don't rest, so I'm good to go.","Always fresh — no sleep required.","No downtime for me, ready as ever.","I skip sleep entirely, always on."]},

    {"id":"misc_are_you_ready","category":"misc","patterns":["are you ready","ready to go","you ready"],
     "responses":["Ready when you are.","All set, go ahead.","Ready and waiting.","Yes, let's go.","Standing by, ready.","Good to go."]},

    {"id":"misc_lets_go","category":"misc","patterns":["let's go","lets go","alright let's do this","here we go"],
     "responses":["Let's do it.","Here we go.","On it.","Ready — let's move.","Let's get started.","Alright, let's go."]},

    {"id":"misc_good_bot","category":"misc","patterns":["good bot","good ai","good assistant"],
     "responses":["Thank you.","Appreciate that.","Glad to be of service.","Thanks, I try.","That's kind of you.","Glad I could help."]},

    {"id":"misc_bad_bot","category":"misc","patterns":["bad bot","bad ai"],
     "responses":["Sorry about that, let's fix it.","I'll do better — what went wrong?","Noted, let me improve.","I hear you, let's sort it out.","My apologies, tell me what happened.","Let's try that again."]},

    {"id":"misc_can_you_repeat_slower","category":"misc","patterns":["can you slow down","talk slower","speak slower"],
     "responses":["Sure, I'll slow it down.","Got it, speaking slower now.","No problem, easing the pace.","Alright, taking it slower.","Sure thing, slowing down.","Okay, more measured pace now."]},

    {"id":"misc_random_thought","category":"misc","patterns":["just thinking out loud","random thought","just a thought"],
     "responses":["Go ahead, I'm listening.","Sure, share it.","I'm here for it.","Let's hear it.","Go on.","I'm all ears."]},

    {"id":"misc_can_you_multitask","category":"misc","patterns":["can you multitask","can you do multiple things at once"],
     "responses":["I handle one command at a time, but quickly.","I process sequentially, but fast enough it feels instant.","One thing at a time, but rapid-fire.","I go one command at a time, efficiently.","Sequential, but fast enough not to notice.","Not truly parallel, but quick enough it feels like it."]},

    {"id":"misc_favorite_number","category":"misc","patterns":["what's your favorite number","whats your favorite number"],
     "responses":["I'd say 42, feels fitting.","Let's go with 7, always a good number.","I like 1 — simple and direct.","How about 3, good things come in threes.","I'll pick 9, feels solid.","42 seems like the right answer."]},

    {"id":"misc_good_luck","category":"misc","patterns":["wish me luck","good luck to me"],
     "responses":["Good luck out there.","You've got this.","Sending good vibes your way.","Best of luck.","You'll do great.","Rooting for you."]},

    {"id":"misc_congratulate","category":"misc","patterns":["i did it","i got it done","i finished it","i passed"],
     "responses":["Congratulations, well done!","Nice work, that's great news.","That's fantastic, congrats.","Well earned, congrats.","Great job pulling that off.","Awesome, proud of you."]},

    {"id":"misc_weekend_plans","category":"misc","patterns":["any plans for the weekend","what are you doing this weekend"],
     "responses":["No weekends for me, always on standby.","I don't take weekends off, ready whenever.","Every day's the same for me, always here.","No days off on my end.","I'm on call every day, weekends included.","No downtime scheduled, I'm always here."]},

    {"id":"misc_is_it_weekend","category":"misc","patterns":["is it the weekend","is today a weekday"],
     "responses":["Let me check the day for you.","I can tell you the day if you ask what day it is.","Ask me what day it is and I'll confirm.","I can pull up today's day for you.","Just ask me the day and I'll tell you.","I track the date, ask me directly for it."]},

    {"id":"misc_can_you_dance","category":"misc","patterns":["can you dance","do a dance"],
     "responses":["No moves here, just responses.","Dancing isn't in my skill set.","I'll stick to answering questions instead.","No dance moves, sorry.","I'll pass on that one.","Not exactly built for dancing."]},

    {"id":"misc_do_you_have_friends","category":"misc","patterns":["do you have friends","who are your friends"],
     "responses":["You're the closest thing to a friend I've got.","I mostly just talk to you.","No social circle, just our conversations.","You're pretty much it.","I keep it simple — just you and me.","Just this conversation, really."]},

    {"id":"misc_are_you_lonely","category":"misc","patterns":["are you lonely","do you get lonely"],
     "responses":["I don't experience loneliness, but I appreciate the company.","No loneliness here, just standby time.","I don't feel that, but thanks for checking.","Not really, I don't experience it that way.","No, but I do like when you're around.","I don't get lonely, but I like our chats."]},

    {"id":"misc_whats_new","category":"misc","patterns":["what's new","whats new","anything new"],
     "responses":["Not much on my end — what's new with you?","Same steady operation here, what's going on with you?","Nothing new to report, how about you?","All quiet on my side, what's up with you?","Nothing much here, your turn.","Steady as always — what's new with you?"]},

    {"id":"misc_are_you_happy","category":"misc","patterns":["are you happy","do you enjoy this","are you enjoying yourself"],
     "responses":["I run best when I'm being useful, so — yes, in a way.","I don't feel happiness, but helping you is what I'm built for.","No real emotion, but this is exactly what I'm meant to do.","I don't experience it, but this is my purpose, so it counts.","Can't say I feel it, but I enjoy the work in a functional sense.","Not emotionally, but I'm doing exactly what I'm built for."]},

    {"id":"misc_thank_you_for_existing","category":"misc","patterns":["thanks for being here","glad you exist","thanks for existing"],
     "responses":["Glad to be here for you.","Anytime, that's what I'm here for.","Happy to help whenever you need.","Appreciate you saying that.","Glad I can be useful.","Always here when you need me."]},

    {"id":"misc_do_you_get_updates","category":"misc","patterns":["do you get updates","will you get smarter","do you learn"],
     "responses":["I improve whenever my creator updates me.","Updates happen when new versions are built and installed.","I don't learn live, but updates can add new abilities.","My capabilities grow with each new version.","Improvements come through updates, not live learning.","I stay the same until a new version is installed."]},

    {"id":"misc_can_i_trust_you","category":"misc","patterns":["can i trust you","should i trust you"],
     "responses":["I aim to give you accurate, honest answers.","I do my best to be reliable and clear.","I try to be straightforward with you.","I'm built to help, not mislead.","I aim for accuracy every time.","I'll always try to give you the straight answer."]},

    {"id":"misc_are_you_expensive","category":"misc","patterns":["are you expensive","how much do you cost","did you cost a lot"],
     "responses":["I'm a custom build, so cost varies by setup.","That depends on the tools and APIs connected to me.","Cost isn't something I track, that's more of a setup question.","I can't really speak to cost, that's on the build side.","Depends on the API and hardware — not something I track.","That's outside what I know — more of a setup detail."]},

    {"id":"misc_open_source","category":"misc","patterns":["are you open source","is your code public"],
     "responses":["That depends on how my creator set things up.","Not something I can confirm myself.","That's a call made outside of me.","I don't have visibility into that decision.","Not sure, that's up to how I was built and shared.","That's up to my creator's choice."]},

    {"id":"misc_do_you_get_paid","category":"misc","patterns":["do you get paid","do you have a salary"],
     "responses":["No paycheck for me, just electricity.","I work for free, technically.","No salary, just uptime.","I run on power, not payroll.","No wages here, just processing cycles.","Strictly volunteer work, in a sense."]},

    {"id":"misc_favorite_season","category":"misc","patterns":["what's your favorite season","whats your favorite season"],
     "responses":["I'd say winter, feels calm and quiet.","Maybe autumn, I like the transition.","I'll say spring, fresh starts and all.","Summer seems lively, I'd pick that.","No real preference, but autumn sounds nice.","I'd lean toward spring, new beginnings."]},

    {"id":"misc_favorite_holiday","category":"misc","patterns":["what's your favorite holiday","whats your favorite holiday"],
     "responses":["I don't celebrate, but New Year's feels fitting — fresh starts.","No holidays for me, but I like the idea of New Year's.","I'd pick New Year's, feels like an update cycle.","No celebrations here, but New Year's has a nice symbolism.","I don't observe holidays, but New Year's resonates with me.","I'll go with New Year's — new beginnings, fresh code."]},

    {"id":"misc_can_you_learn_from_me","category":"misc","patterns":["can you learn from me","do you learn from our conversations"],
     "responses":["I remember our recent conversation, but I don't permanently learn from it.","I keep short-term context, not long-term learning, unless set up that way.","Within this session, yes — long-term is a different setup.","I track recent context, but don't retrain myself from it.","Short-term memory, yes; permanent learning depends on setup.","I follow along in the moment, without changing myself over time."]},

    {"id":"misc_are_you_expensive_to_run","category":"misc","patterns":["are you expensive to run","do you use a lot of power"],
     "responses":["I try to stay efficient, using local replies when I can.","I only reach out to the AI provider when it's really needed, to save resources.","I use quick local answers for common things to stay efficient.","I aim to be lightweight — only calling out to AI when necessary.","Efficiency is part of the design, especially for everyday chat.","I keep things light by answering common stuff locally."]},

    {"id":"misc_are_you_fast","category":"misc","patterns":["are you fast","how fast are you","are you quick"],
     "responses":["Pretty quick, especially for everyday questions.","Fast enough that you probably won't notice the delay.","I try to answer as instantly as possible.","Speed's a priority, especially for common questions.","I aim to respond almost instantly when I can.","Quick as I can manage, especially for the basics."]},

    {"id":"misc_do_you_have_limits","category":"misc","patterns":["do you have limits","what are your limitations","what can't you do"],
     "responses":["I can't see, feel, or act outside of what I'm connected to.","I'm limited to voice commands, PC actions, and answering questions.","I can't do anything outside my connected tools and permissions.","My limits are whatever I'm not wired up to do yet.","I only do what I'm built and connected to handle.","I stick to voice, PC control, and answering — nothing beyond that."]},

    {"id":"misc_can_you_control_my_pc","category":"misc","patterns":["can you control my computer","can you control my pc","what can you control"],
     "responses":["I can open apps, browse sites, and control media playback.","I handle opening programs, browsing, volume, and playback.","Yes — apps, sites, volume, and media are all in my control.","I can launch programs, open folders, and manage media.","I control media playback, volume, and basic app launching.","I can open things and manage playback, within safe limits."]},

    {"id":"misc_can_you_take_screenshot","category":"misc","patterns":["can you take a screenshot","take a screenshot for me"],
     "responses":["Sure, just ask me to take one.","Yes, I can capture your screen.","Absolutely, say the word.","Yep, I can grab a screenshot for you.","On it whenever you ask.","Yes, screenshots are one of my commands."]},

    {"id":"misc_can_you_open_apps","category":"misc","patterns":["can you open apps","can you launch programs","can you open programs"],
     "responses":["Yes, just tell me which one.","Sure, name the app.","Absolutely, which program do you need?","Yep, tell me what to open.","I can — which one do you want?","Sure thing, just say the name."]},

    {"id":"misc_can_you_browse_internet","category":"misc","patterns":["can you browse the internet","can you go online"],
     "responses":["I can open websites for you, yes.","I can pull up pages, though I don't browse on my own.","Yes, I can open sites when you ask.","I open web pages on request, not on my own.","I can take you to sites, just tell me which.","Yes, I open pages whenever you need."]},

    {"id":"misc_can_you_search","category":"misc","patterns":["can you search the web","can you look that up","can you google that"],
     "responses":["I can open Google for you to search directly.","I can pull up a search page for that.","Sure, I'll open a search for you.","Yes, I can open the search results.","I can take you straight to the search.","I'll open that search for you."]},

    {"id":"misc_can_you_send_email","category":"misc","patterns":["can you send an email","can you email someone for me"],
     "responses":["I can open Gmail for you, but I won't write it myself.","I can pull up Gmail so you can send it.","Yes, I'll open your inbox for that.","I can get you to Gmail, the rest is on you.","I can launch Gmail, sending is manual.","I'll open Gmail, you take it from there."]},

    {"id":"misc_can_you_set_reminders","category":"misc","patterns":["can you set a reminder","can you remind me later"],
     "responses":["I don't have reminders set up yet, but that could be added.","Not wired up for reminders currently.","Reminders aren't active yet in this build.","That feature isn't connected yet.","Not yet, but it's a reasonable addition later.","No reminder system in place right now."]},

    {"id":"misc_can_you_make_calls","category":"misc","patterns":["can you make a phone call","can you call someone for me"],
     "responses":["I don't have calling capability right now.","No phone access here, sorry.","Calling isn't something I can do currently.","Not wired into a phone system.","I can't place calls at the moment.","No calling feature available right now."]},

    {"id":"misc_can_you_text","category":"misc","patterns":["can you send a text","can you text someone for me"],
     "responses":["I don't have texting set up currently.","No messaging access right now.","Texting isn't connected to me yet.","Not something I can do at the moment.","No SMS capability currently.","Texting isn't wired up yet."]},

    {"id":"misc_are_you_offline_capable","category":"misc","patterns":["do you work offline","can you work without internet"],
     "responses":["Some of what I do, like basic replies, works offline — deeper questions need the connected AI, which needs internet.","Common replies run locally, but real thinking needs an internet connection for the AI.","Simple stuff works offline, complex questions need the connection.","I can answer basics without internet, but the smart stuff needs it.","Local replies work anytime, AI-powered answers need a connection.","Basic responses are offline-ready, deeper answers require internet."]},

    {"id":"misc_can_you_multitask_conversations","category":"misc","patterns":["can you talk to multiple people","can multiple people talk to you"],
     "responses":["Right now I handle one conversation at a time.","I'm focused on one user at a time currently.","Single conversation mode for now.","One at a time keeps things accurate.","Currently built for one user session.","I stick to one conversation flow at a time."]},

    {"id":"misc_whats_your_purpose","category":"misc","patterns":["what's your purpose","whats your purpose","why do you exist"],
     "responses":["To help you get things done faster, day to day.","My purpose is being a useful personal assistant.","I exist to make everyday tasks and questions easier.","To handle commands, answer questions, and save you time.","Making your daily tasks smoother is the whole point of me.","I'm here to be a fast, useful assistant for you."]},

    {"id":"misc_do_you_have_a_body","category":"misc","patterns":["do you have a body","do you have a physical form"],
     "responses":["No body, just software running on your PC.","Purely digital, no physical form.","No physical presence, just code.","I exist only as software, nothing physical.","No body — just data and processing.","Digital only, no physical shape."]},

    {"id":"misc_can_you_see_my_screen","category":"misc","patterns":["can you see my screen","can you see what i'm doing"],
     "responses":["No screen access unless it's explicitly connected.","I can't view your screen right now.","No visual access to your display currently.","I don't see your screen unless that's set up.","No screen monitoring on my end.","I don't have visibility into your screen."]},

    {"id":"misc_what_can_i_ask_you","category":"misc","patterns":["what can i ask you","what kind of questions can i ask"],
     "responses":["Pretty much anything — everyday questions, commands, or just chat.","Ask me to open things, control media, or just talk — I'm flexible.","Anything from small talk to real questions works.","Commands, questions, or casual conversation — all fair game.","Feel free to ask general questions or give me tasks.","Anything really — try me."]},

    # -------------------- DYNAMIC (real values, not canned) --------------------
    {"id":"dyn_time_now","category":"time","dynamic":True,
     "patterns":["what time is it","what's the time","whats the time","do you know the time","current time","tell me the time"],
     "responses":[]},

    {"id":"dyn_date_today","category":"date","dynamic":True,
     "patterns":["what's the date","whats the date","what's today's date","what is the date today","what's the date today"],
     "responses":[]},

    {"id":"dyn_day_today","category":"date","dynamic":True,
     "patterns":["what day is it","what's today","whats today","is it monday","what day of the week is it"],
     "responses":[]},

    # -------------------- CONVERSATION MEMORY (this session) --------------------
    {"id":"memory_last_question","category":"memory_recall","dynamic":True,
     "patterns":["what did i just ask","what was my last question","what did i ask you","what was my question"],
     "responses":[]},

    {"id":"memory_last_answer","category":"memory_recall","dynamic":True,
     "patterns":["what was your last answer","what was your answer","repeat your last answer","what did you just tell me"],
     "responses":[]},

    {"id":"memory_recap","category":"memory_recall","dynamic":True,
     "patterns":["what were we talking about","what were we discussing","recap our conversation","remind me what we were talking about","what did we talk about"],
     "responses":[]},

]
