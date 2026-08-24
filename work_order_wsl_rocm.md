# 작업지시서: WSL2 ROCm/HIP 환경에 Zipformer + MOSS-TTS-Nano 설치

## 배경
- Whisper large-v3, CosyVoice3는 설치 완료 (CUDA 기준으로 가정)
- 추가 설치 대상: 한국어 Zipformer(sherpa-onnx), MOSS-TTS-Nano-100M
- 이 두 모델은 **CUDA가 아니라 ROCm/HIP 환경**(AMD GPU)에서 설치·검증할 것
- 실행 환경: Windows 11 + WSL2 (Ubuntu 22.04 또는 24.04)

## 전제 조건 확인 (작업 시작 전 반드시 확인)
1. AMD GPU가 ROCm WSL 지원 목록에 포함되는지 확인할 것 (RDNA3/RDNA4 데스크톱 GPU 또는 Ryzen AI Strix/Strix Halo 계열 중심으로 지원됨. 구형 GPU는 미지원 가능성 높음)
2. Windows 11 빌드 버전이 WSL2 GPU 패스스루를 지원하는지 확인
3. `wsl --version`으로 WSL 커널이 최신인지 확인
4. 현재 CUDA용으로 설치된 PyTorch/onnxruntime과 ROCm용 패키지가 **동일 Python 환경에 섞이지 않도록** 반드시 별도 venv 또는 별도 WSL distro/컨테이너로 분리할 것

## 1단계: Windows 호스트 측 드라이버 설치
1. AMD Adrenalin Edition (WSL2 지원 버전, 26.2.2 이상) 드라이버를 Windows에 설치
   - 다운로드: AMD 공식 드라이버 페이지에서 "WSL2용" 표기된 버전 선택
   - 주의: 이 드라이버는 WSL 내부가 아니라 **Windows 호스트에 설치**하는 것
2. 설치 후 Windows 재부팅
3. Windows PowerShell에서 GPU 인식 확인: `dxdiag` 실행 후 디스플레이 탭에서 GPU 확인

## 2단계: WSL2 준비
```powershell
# PowerShell (관리자 권한)
wsl --update
wsl --list --verbose
```
- Ubuntu 22.04 또는 24.04 배포판인지 확인 (ROCm WSL 공식 지원 배포판)
- 다른 배포판이면 새로 설치 권장: `wsl --install -d Ubuntu-24.04`

## 3단계: WSL 내부 - ROCDXG(librocdxg) + ROCm 설치
```bash
# WSL Ubuntu 내부에서 실행
sudo apt update && sudo apt upgrade -y

# librocdxg 설치 (공식 GitHub Quickstart 절차 따름)
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
# README의 Quickstart 스크립트 실행 (버전에 따라 스크립트명이 다를 수 있으니 최신 README 확인 필수)
cat README.md   # 설치 스크립트 경로 확인 후 실행

# ROCm 7.2.1 이상 설치 (Radeon용 설치 가이드 절차 따름)
# 공식 문서: https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html
# 위 문서의 amdgpu-install 스크립트를 WSL 배포판(Ubuntu 22.04/24.04)에 맞춰 실행
```

**중요**: librocdxg와 ROCm 설치 스크립트는 버전이 자주 바뀌므로, 반드시 작업 시점의 공식 문서(`rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/`)에서 최신 스크립트를 확인하고 그대로 따를 것. 이 문서에 적힌 특정 명령어를 맹신하지 말고 문서 페이지의 최신 버전과 대조할 것.

## 4단계: 설치 검증
```bash
rocminfo                 # GPU 인식 여부 확인, "Agent" 항목에 AMD GPU가 떠야 함
rocm-smi                 # GPU 사용률/메모리 확인
```
- `rocminfo`에서 GPU가 안 잡히면 2~3단계를 재점검할 것 (Windows 드라이버 미설치, WSL 커널 버전 문제가 대부분의 원인)

## 5단계: MOSS-TTS-Nano용 PyTorch(ROCm) 설치
```bash
# 전용 venv 생성 (기존 CUDA venv와 절대 혼용 금지)
python3 -m venv ~/venv-moss-rocm
source ~/venv-moss-rocm/bin/activate

# ROCm 빌드 PyTorch 설치 (버전 번호는 설치된 ROCm 버전과 일치시킬 것)
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

# 설치 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# ROCm 환경에서는 torch.cuda API가 HIP 백엔드로 매핑되어 동작함 (정상)
```

이후 MOSS-TTS-Nano 설치:
```bash
git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git
cd MOSS-TTS-Nano
pip install -r requirements.txt
# 한국어 문장으로 동작 테스트 (README의 generate 명령 참고)
```

## 6단계: Zipformer(sherpa-onnx)용 onnxruntime-ROCm 설치
```bash
python3 -m venv ~/venv-zipformer-rocm
source ~/venv-zipformer-rocm/bin/activate

pip install sherpa-onnx
# onnxruntime의 ROCm/MIGraphX 실행 프로바이더 설치 시도
pip install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
```

**주의 (반드시 검증 필요)**: sherpa-onnx 공식 배포판은 CPU/CUDA 중심으로 빌드되어 있고, ROCm 실행 프로바이더(MIGraphX) 공식 지원 여부가 불확실함. 다음 순서로 확인할 것:
1. `pip install onnxruntime-rocm` 설치 시도 → 실패하면 sherpa-onnx가 ROCm을 지원하지 않는 것으로 판단
2. 실패 시, **Zipformer는 애초에 모바일/CPU 온디바이스용 모델**이므로 ROCm/GPU 가속 없이 CPU 모드로 테스트하는 것으로 대체할 것 (기능 검증 목적이면 CPU로도 충분)
3. GPU 가속이 꼭 필요하면 sherpa-onnx GitHub 이슈에서 "ROCm" "MIGraphX" 키워드로 지원 현황을 먼저 확인 후 진행

## 7단계: 최종 검증 체크리스트
- [ ] `rocminfo`에 AMD GPU가 정상 인식됨
- [ ] PyTorch에서 `torch.cuda.is_available() == True` (ROCm/HIP 백엔드)
- [ ] MOSS-TTS-Nano로 한국어 문장 음성 생성 성공, 결과 wav 파일 청취 확인
- [ ] Zipformer(sherpa-onnx)로 한국어 wav 파일 STT 테스트 성공 (CPU 또는 ROCm)
- [ ] 기존 CUDA 기반 Whisper large-v3 / CosyVoice3 환경과 패키지 충돌 없음 확인 (별도 venv로 분리되어 있는지 재확인)

## 산출물
- 각 단계 실행 로그
- `rocminfo`, `rocm-smi` 출력 결과
- MOSS-TTS-Nano 한국어 생성 샘플 wav
- Zipformer 한국어 STT 테스트 결과 (인식된 텍스트)
- 설치 중 발생한 에러 메시지 전문 (있는 경우)
