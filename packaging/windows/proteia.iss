; SPDX-License-Identifier: Apache-2.0
;
; Inno Setup script for Proteia's Windows installer (ADR 0003). build.py compiles
; it with Inno Setup 7.1.0 and passes the defines checked below:
;
;   ISCC.exe /DAppName=... /DExeName=... /DAppVersion=... /DNumericVersion=...
;            /DAppPublisher=... /DAppURL=... /DDistDir=<the PyInstaller folder>
;            /O<output folder> proteia.iss
;
; What it does:
; - installs the PyInstaller folder (with LICENSE.txt and THIRD_PARTY_NOTICES.txt)
;   per user by default, into %LOCALAPPDATA%\Programs\Proteia, without an
;   administrator prompt; an administrator may choose a per-machine install
;   (the dialog, or /ALLUSERS);
; - adds a Start Menu shortcut, and a desktop shortcut when asked;
; - before it replaces files, and before an uninstall removes any (after the
;   user has confirmed it), stops the Proteia this user runs: it sends Quit
;   (POST /api/quit) with the token from the launcher's instance.json, as the
;   page's Quit button does, so Proteia saves first. It stops with a message
;   when Proteia cannot save (409), does not answer, or does not end, and when
;   the installed Proteia.exe is still in use, as while a Proteia Quit could
;   not reach runs it (another user's, say): nothing is deleted from under a
;   running Proteia;
; - an uninstall removes the program files and the launcher's own files in the
;   uninstalling user's state folder (instance.lock, instance.json,
;   open-proteia.html), keeps whatever else that folder holds (the session log
;   in its logs folder, proteia.web.logs, #137), and never touches the projects
;   in Documents\Proteia.
; The state folder is found as the launcher finds it (proteia.web.launch.state_dir):
; %LOCALAPPDATA%\Proteia, from the environment variable first.

#define RequiredInnoSetup "7.1.0"
#if DecodeVer(Ver, 3) != RequiredInnoSetup
  #error This script is built with Inno Setup 7.1.0 (ADR 0003: the build pins it)
#endif
#ifndef AppName
  #error AppName is not defined: build the installer with packaging/windows/build.py
#endif
#ifndef ExeName
  #error ExeName is not defined: build the installer with packaging/windows/build.py
#endif
#ifndef AppVersion
  #error AppVersion is not defined: build the installer with packaging/windows/build.py
#endif
#ifndef NumericVersion
  #error NumericVersion is not defined: build the installer with packaging/windows/build.py
#endif
#ifndef AppPublisher
  #error AppPublisher is not defined: build the installer with packaging/windows/build.py
#endif
#ifndef AppURL
  #error AppURL is not defined: build the installer with packaging/windows/build.py
#endif
#ifndef DistDir
  #error DistDir is not defined: build the installer with packaging/windows/build.py
#endif

[Setup]
; The application ID never changes: upgrades and the uninstall entry depend on it.
AppId={{B32AD498-6981-42C9-9528-00A05682ECD0}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
VersionInfoVersion={#NumericVersion}
VersionInfoTextVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Setup
; Per user by default ({autopf} is %LOCALAPPDATA%\Programs then); an administrator
; may install for all users instead (the dialog, or /ALLUSERS).
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
; Windows 10 and later: the bundle leaves out the Universal CRT they provide.
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputBaseFilename={#AppName}-{#AppVersion}-setup
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#ExeName}
; [Code] refuses to go on while the installed Proteia.exe is in use (PrepareToInstall
; runs before this check); the Restart Manager then handles only other programs
; holding the files, and is not asked to start them again.
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Tasks]
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An upgrade replaces the whole bundle: a module or package metadata the new
; version no longer ships must not stay behind to be loaded or read.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  StateFolderName = 'Proteia';
  { The launcher's files in the state folder (proteia.web.launch). }
  LockFileName = 'instance.lock';
  InstanceFileName = 'instance.json';
  RedirectFileName = 'open-proteia.html';
  SYNCHRONIZE = $00100000;
  PROCESS_QUERY_LIMITED_INFORMATION = $00001000;
  WAIT_OBJECT_0 = 0;
  QuitWaitMs = 30000;
  { How long Proteia.exe may stay busy after its process has ended. }
  InUseWaitMs = 5000;
  InUseStepMs = 250;
  { WinHttpRequest's HTTPREQUEST_PROXYSETTING_DIRECT: the loopback port, no proxy. }
  ProxyDirect = 1;

function OpenProcess(DesiredAccess: DWORD; InheritHandle: BOOL; ProcessId: DWORD): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function WaitForSingleObject(Handle: THandle; Milliseconds: DWORD): DWORD;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function CloseHandle(Handle: THandle): BOOL;
  external 'CloseHandle@kernel32.dll stdcall';
function QueryFullProcessImageName(Process: THandle; Flags: DWORD; ExeName: String;
  var Size: DWORD): BOOL;
  external 'QueryFullProcessImageNameW@kernel32.dll stdcall';

{ The launcher's state folder: %LOCALAPPDATA%\Proteia, the variable first, as
  proteia.web.launch.state_dir() finds it. }
function StateFolder: String;
begin
  Result := GetEnv('LOCALAPPDATA');
  if Result = '' then
    Result := ExpandConstant('{localappdata}');
  Result := AddBackslash(Result) + StateFolderName;
end;

{ Where the value of "Key" starts in Json, the flat object json.dumps wrote; 0
  when the key is missing. }
function JsonValueAt(const Json, Key: String): Integer;
var
  P: Integer;
begin
  Result := 0;
  P := Pos('"' + Key + '"', Json);
  if P = 0 then
    Exit;
  P := P + Length(Key) + 2;
  while (P <= Length(Json)) and ((Json[P] = ' ') or (Json[P] = ':')) do
    P := P + 1;
  if P <= Length(Json) then
    Result := P;
end;

function JsonDigits(const Json, Key: String): String;
var
  P: Integer;
begin
  Result := '';
  P := JsonValueAt(Json, Key);
  if P = 0 then
    Exit;
  while (P <= Length(Json)) and (Json[P] >= '0') and (Json[P] <= '9') do
  begin
    Result := Result + Json[P];
    P := P + 1;
  end;
end;

function JsonText(const Json, Key: String): String;
var
  P: Integer;
begin
  Result := '';
  P := JsonValueAt(Json, Key);
  if (P = 0) or (Json[P] <> '"') then
    Exit;
  P := P + 1;
  while (P <= Length(Json)) and (Json[P] <> '"') do
  begin
    Result := Result + Json[P];
    P := P + 1;
  end;
end;

{ A token as the launcher makes them: 32 to 128 URL-safe characters
  (proteia.web.server.TOKEN_PATTERN). }
function IsToken(const S: String): Boolean;
var
  I: Integer;
  C: Char;
begin
  Result := (Length(S) >= 32) and (Length(S) <= 128);
  for I := 1 to Length(S) do
  begin
    C := S[I];
    if not (((C >= 'A') and (C <= 'Z')) or ((C >= 'a') and (C <= 'z'))
        or ((C >= '0') and (C <= '9')) or (C = '-') or (C = '_')) then
      Result := False;
  end;
end;

{ The file name of the program Process runs; '' when Windows does not say. }
function ProcessExeName(Process: THandle): String;
var
  Path: String;
  Size: DWORD;
begin
  Result := '';
  Size := 1024;
  SetLength(Path, Size);
  if QueryFullProcessImageName(Process, 0, Path, Size) then
    Result := ExtractFileName(Copy(Path, 1, Size));
end;

{ Stop the Proteia this user runs, if one runs: send Quit with the token from
  instance.json, as the page's Quit button does (Proteia saves first and answers
  409 when it cannot), then wait for its process to end. Returns '' when no
  Proteia runs any more, or when instance.json is stale (its process has ended,
  or its id now belongs to another program); else why Proteia could not be
  stopped, which includes a Proteia that does not answer Quit at all. }
function StopRunningProteia: String;
var
  Json: AnsiString;
  Text, Token, Name: String;
  Pid, Port, Status: Integer;
  Process: THandle;
  Request: Variant;
begin
  Result := '';
  if not LoadStringFromFile(StateFolder + '\' + InstanceFileName, Json) then
    Exit;
  Text := String(Json);
  Pid := StrToIntDef(JsonDigits(Text, 'pid'), 0);
  Port := StrToIntDef(JsonDigits(Text, 'port'), 0);
  Token := JsonText(Text, 'token');
  if (Pid <= 0) or (Port <= 0) or (Port > 65535) or not IsToken(Token) then
  begin
    Log('Proteia: ' + InstanceFileName + ' is not one the launcher wrote; nothing to stop.');
    Exit;
  end;
  Process := OpenProcess(SYNCHRONIZE or PROCESS_QUERY_LIMITED_INFORMATION, False, Pid);
  if Process = 0 then
  begin
    Log('Proteia: the process ' + InstanceFileName + ' names has ended.');
    Exit;
  end;
  try
    Status := 0;
    try
      Request := CreateOleObject('WinHttp.WinHttpRequest.5.1');
      Request.SetProxy(ProxyDirect);
      Request.SetTimeouts(2000, 2000, 5000, 30000);
      Request.Open('POST', 'http://127.0.0.1:' + IntToStr(Port) + '/api/quit', False);
      Request.SetRequestHeader('Authorization', 'Bearer ' + Token);
      Request.Send('');
      Status := Request.Status;
    except
      Log('Proteia: Quit could not be sent: ' + GetExceptionMessage);
    end;
    Log('Proteia: Quit answered ' + IntToStr(Status) + '.');
    if Status = 0 then
    begin
      { No answer: a hung or unreachable Proteia, or a stale instance.json whose
        process id Windows has given to another program. }
      Name := ProcessExeName(Process);
      if WaitForSingleObject(Process, 0) = WAIT_OBJECT_0 then
        Log('Proteia: the process ' + InstanceFileName + ' names has ended.')
      else if CompareText(Name, '{#ExeName}') = 0 then
        Result := '{#AppName} is running but does not answer. Quit it with Quit on its ' +
          'page, or close its window if the page does not respond, and try again.'
      else if Name = '' then
        Log('Proteia: the program process ' + IntToStr(Pid) + ' runs is not known.')
      else
        Log('Proteia: process ' + IntToStr(Pid) + ' runs ' + Name + ', not {#ExeName}.');
    end
    else if Status = 409 then
      Result := '{#AppName} is running and could not save the open project. Open ' +
        '{#AppName}, deal with what it reports, quit it, and try again.'
    else if Status <> 202 then
      Result := '{#AppName} is running and did not accept Quit (HTTP ' + IntToStr(Status) +
        '). Quit {#AppName} with Quit on its page, and try again.'
    else if WaitForSingleObject(Process, QuitWaitMs) <> WAIT_OBJECT_0 then
      Result := '{#AppName} is still running after Quit. Close it, and try again.';
  finally
    CloseHandle(Process);
  end;
end;

{ Whether Exe can be opened for writing: not while a process runs it. }
function CanOpenForWriting(const Exe: String; var Error: String): Boolean;
var
  Stream: TFileStream;
begin
  Result := True;
  try
    Stream := TFileStream.Create(Exe, fmOpenWrite or fmShareDenyNone);
    Stream.Free;
  except
    Result := False;
    Error := GetExceptionMessage;
  end;
end;

{ Why the installed program cannot be replaced or removed now: '' unless the
  installed Proteia.exe is there and busy, as while any Proteia runs it (this
  user's, one Quit could not reach, or another user's). Checked before anything
  is deleted, so that [InstallDelete] or the uninstaller never removes the
  files of a running Proteia; a Proteia that has just quit gets a moment. }
function ProgramInUse: String;
var
  Exe, Error: String;
  Waited: Integer;
begin
  Result := '';
  Exe := ExpandConstant('{app}\{#ExeName}');
  if not FileExists(Exe) then
    Exit;
  Waited := 0;
  while not CanOpenForWriting(Exe, Error) do
  begin
    if Waited >= InUseWaitMs then
    begin
      Log('Proteia: ' + Exe + ' is in use: ' + Error);
      Result := Exe + ' is in use: {#AppName} is still running (for this or another ' +
        'user). Quit it with Quit on its page, and try again.';
      Exit;
    end;
    Sleep(InUseStepMs);
    Waited := Waited + InUseStepMs;
  end;
end;

{ Stop this user's Proteia, then make sure no Proteia runs the installed
  program: '' when the files may go, else what the user has to do. }
function ProteiaProblem: String;
begin
  Result := StopRunningProteia;
  if Result = '' then
    Result := ProgramInUse;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := ProteiaProblem;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Problem, Folder: String;
begin
  if CurUninstallStep = usAppMutexCheck then
  begin
    { After the user has confirmed the uninstall, before anything is removed. }
    Problem := ProteiaProblem;
    if Problem <> '' then
    begin
      SuppressibleMsgBox(Problem, mbError, MB_OK, IDOK);
      Abort;
    end;
  end
  else if CurUninstallStep = usPostUninstall then
  begin
    Folder := StateFolder;
    DeleteFile(Folder + '\' + LockFileName);
    DeleteFile(Folder + '\' + InstanceFileName);
    DeleteFile(Folder + '\' + RedirectFileName);
    { Only when nothing else is left in it: the logs folder (the session log) stays. }
    RemoveDir(Folder);
  end;
end;
