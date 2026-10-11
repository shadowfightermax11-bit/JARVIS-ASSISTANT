J.A.R.V.I.S. Companion — corrected build package

WHAT IS INCLUDED (SOURCE / BUILDER PACKAGE — NOT A PREBUILT EXE)
- edith_gemini_voice.py: latest Companion source supplied in the conversation (v3.0.0)
- local_responses.py: latest local-response layer, with the Windows date-format issue corrected
- JARVIS-Companion.iss: installer definition; it does NOT ask for an API provider or API key
- requirements.txt: runtime/build Python packages
- build_installer.bat: creates the Windows app folder and installer

WEBSITE / COMPANION INTEGRATION
- The website's hand-tracking engine lives in the JARVIS HTML and uses browser camera permission + MediaPipe.
- The Python Companion exposes localhost bridge endpoints on 127.0.0.1:8765, including /status, /state, /config, and /media.
- The website sends provider settings to /config. Provider/API-key setup stays in the website, as requested.
- Browser bridge commands are restricted to the GitHub Pages origin https://shadowfightermax11-bit.github.io.
- The packaged Companion runs without a console window; diagnostics are written to %APPDATA%\JARVIS\companion.log.

BUILD ON WINDOWS
1. Extract this ZIP to a normal folder.
2. Install Python 3.11+ and Inno Setup 6 on the build PC.
3. Ensure the Windows Python launcher `py` and Inno Setup compiler `ISCC.exe` are available on PATH.
4. Double-click build_installer.bat.
5. On success, find output\JARVIS-Setup.exe.
6. The builder outputs output\JARVIS-Setup.exe. For a public download, publish it as the JARVIS-Setup.exe asset on a GitHub Release. The supplied .github/workflows/build-installer-release.yml can build it on GitHub-hosted Windows runners when copied into the repository root: push a tag named jarvis-v3.0.1 (use a new version tag for each release) to build and publish automatically. The website cannot silently install Windows software; the user must approve/run the downloaded installer once.

NOTES
- This package was statically checked for Python syntax and source/endpoint consistency in this environment. The actual PyInstaller + Inno Setup build must be run on Windows; it was not executed here.
- The first launch may require Windows microphone/camera permissions and internet access for voice/TTS and the browser-based MediaPipe model.
- The startup task is optional and unchecked by default.
