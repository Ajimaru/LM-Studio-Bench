; Owner approval and an owner-supplied vendor EULA are checked by the builder.
; Internal builds have an explicit non-distribution notice instead.
#ifndef BundleDir
  #error BundleDir must be supplied by build_windows.py
#endif
#ifndef VendorEula
  #error VendorEula must be supplied by build_windows.py
#endif
#ifndef AppVersion
  #error AppVersion must be supplied by build_windows.py
#endif
#ifndef SetupName
  #error SetupName must be supplied by build_windows.py
#endif
#ifndef OutputPath
  #error OutputPath must be supplied by build_windows.py
#endif
#ifndef NumericVersion
  #error NumericVersion must be supplied by build_windows.py
#endif

[Setup]
AppId={{AEBA4152-BC29-4F66-B4E5-A43D7D245D63}
AppName=LM Studio Bench
AppVersion={#AppVersion}
AppVerName=LM Studio Bench {#AppVersion}
AppPublisher=Ajimaru
AppPublisherURL=https://github.com/Ajimaru/LM-Studio-Bench
DefaultDirName={localappdata}\Programs\LM-Studio-Bench
DefaultGroupName=LM Studio Bench
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
LicenseFile={#VendorEula}
OutputDir={#OutputPath}
OutputBaseFilename={#SetupName}
VersionInfoVersion={#NumericVersion}
SetupIconFile={#BundleDir}\_internal\assets\icons\lmstudio-bench.ico
UninstallDisplayIcon={app}\LM-Studio-Bench.exe
Compression=lzma2
SolidCompression=yes
CloseApplications=no
RestartApplications=no
SetupLogging=yes
ChangesAssociations=no
ChangesEnvironment=no
Uninstallable=yes

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LM Studio Bench"; Filename: "{app}\LM-Studio-Bench.exe"; \
    WorkingDir: "{app}"

[Code]
function InitializeSetup(): Boolean;
var
  NativeArchitecture: String;
begin
  { x64compatible also permits emulation on ARM64; this release does not. }
  NativeArchitecture := GetEnv('PROCESSOR_ARCHITEW6432');
  if NativeArchitecture = '' then
    NativeArchitecture := GetEnv('PROCESSOR_ARCHITECTURE');
  Result := CompareText(NativeArchitecture, 'AMD64') = 0;
  if not Result then
    MsgBox('This release requires Windows 10/11 x64 (not ARM64).',
      mbError, MB_OK);
end;
