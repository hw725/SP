# CSP 설치마법사(Installer) 배포 및 사용법

## 배포 파일 구성

- `install_wizard_gui.exe` : 설치마법사 실행파일 (dist 폴더)
- (config.json은 exe 내부에서 원격으로만 사용, 별도 배포 불필요)

## 배포/설치 방법

1. `install_wizard_gui.exe`만 배포(또는 릴리즈에 업로드)
2. 사용자는 exe만 다운로드/실행
3. exe는 항상 원격 config.json(예: GitHub)에서 최신 설정을 받아 동작
4. config.json만 변경하면 전체 사용자에 즉시 반영됨

## config.json 관리

- config.json은 반드시 원격(GitHub 등)에만 두고, exe는 항상 해당 URL에서 다운로드
- config.json 예시 및 위치는 install_wizard_gui.py 상단의 CONFIG_URL에서 확인/수정
- config.json을 변경하면 exe 재배포 없이 전체 사용자에 적용

## 빌드 자동화

- `build_installer.bat` 실행 시 PyInstaller로 exe 자동 빌드
- 빌드 산출물: `dist/install_wizard_gui.exe`

## 주의사항

- exe는 반드시 원격 config.json이 접근 가능한 환경에서만 동작
- 로컬 config.json은 무시됨
- config.json이 없거나 다운로드 실패 시 설치마법사 실행 불가

---

최신 config.json 및 배포/설치 문의는 관리자에게 연락 바랍니다.
