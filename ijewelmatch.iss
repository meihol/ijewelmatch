[Setup]
AppId={{8ACDADDE-30FA-4230-BCE3-C84D50727E03}
AppName=StyleLens
AppVersion=1.30
DefaultDirName={pf}\StyleLens
DefaultGroupName=StyleLens
OutputBaseFilename=Setup_StyleLens
Compression=lzma
SolidCompression=yes
; Request admin privileges for installing in Program Files
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "install_python.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_python_server.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "setup_environment.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "check_os.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "run.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "uninstall.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "ijewelmatch.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "base_model.pkl"; DestDir: "{app}"; Flags: ignoreversion
Source: "static\*"; DestDir: "{app}\static"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "templates\*"; DestDir: "{app}\templates"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{cm:UninstallProgram,StyleLens}"; Filename: "{uninstallexe}"
Name: "{group}\Run StyleLens"; Filename: "{app}\run.bat"
Name: "{commondesktop}\StyleLens"; Filename: "{app}\run.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\check_os.bat"; Description: "Perform OS check and run appropriate installation script"; Flags: postinstall skipifsilent runascurrentuser
