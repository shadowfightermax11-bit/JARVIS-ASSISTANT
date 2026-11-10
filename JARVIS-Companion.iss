#define MyAppName "J.A.R.V.I.S. Companion"
#define MyAppVersion "3.0.0"
#define MyAppPublisher "J.A.R.V.I.S."
#define MyAppExeName "JARVIS Companion.exe"

[Setup]
AppId={{A8B7C1A0-4C7F-4E32-9A8C-9D8A4D7F21A1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\JARVIS Companion
DefaultGroupName=J.A.R.V.I.S.
OutputDir=output
OutputBaseFilename=JARVIS-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "startup"; Description: "Start J.A.R.V.I.S. Companion automatically with Windows"; GroupDescription: "Windows startup:"; Flags: unchecked

[Files]
Source: "dist\JARVIS Companion\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\J.A.R.V.I.S. Companion"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\J.A.R.V.I.S. Companion"; Filename: "{app}\{#MyAppExeName}"
Name: "{userstartup}\J.A.R.V.I.S. Companion"; Filename: "{app}\{#MyAppExeName}"; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch J.A.R.V.I.S. Companion"; Flags: nowait postinstall skipifsilent
