# Speech Challenge 2026

전장 및 극한 소음 환경에서의 음성 인식(STT, Speech-to-Text)을 위한 연구/MLOps 표준 기반 Whisper 모델 파인튜닝 프로젝트입니다. 구형 무전기 대역폭 제한(Narrowband) 및 가우시안 노이즈(-5dB ~ 0dB SNR)가 결합된 열화 환경에서 LoRA(PEFT)를 적용해 음성 인식 성능(WER/CER)을 개선합니다.

---

## 📁 프로젝트 구조 (Repository Structure)

```text
Speech-Challenge-2026/
├── configs/                    # 실험 및 하이퍼파라미터 설정 파일
│   └── default_config.yaml     # 모델, 데이터셋, LoRA, 훈련 하이퍼파라미터 기본 설정
│
├── data/                       # 로컬 데이터셋 저장소 (.gitignore 대상)
│
├── checkpoints/                # 모델 가중치 및 학습 체크포인트 (.gitignore 대상)
│
├── notebooks/                  # 탐색 및 분석용 인터랙티브 노트북
│   └── exploration.ipynb       # EDA, 오디오 재생/시각화, 샘플 추론 비교
│
├── src/                        # 핵심 머신러닝 모듈
│   ├── __init__.py
│   ├── data_loader.py          # LibriSpeech 등 오디오 데이터셋 다운로드 및 로드
│   ├── preprocess.py           # 무전 대역폭/노이즈 열화 증강, Whisper 특징 추출 & DataCollator
│   ├── model.py                # Whisper 모델 로드 및 LoRA(PEFT) 설정
│   ├── train.py                # 엔드투엔드 학습/평가 파이프라인 및 Config 연동 메인 진입점
│   ├── evaluate.py             # 독립 평가 스크립트 (WER, CER 지표 산출 및 리포트 저장)
│   └── utils.py                # 재현성을 위한 시드 고정, 로깅 설정, YAML 파서
│
├── tests/                      # 단위 테스트 (Unit Tests)
│   ├── __init__.py
│   └── test_preprocess.py     # 전처리 및 열화 텐서 차원 정합성 검증 테스트
│
├── requirements.txt            # 프로젝트 의존성 패키지 목록
├── .gitignore                  # 대용량 데이터, 모델 체크포인트, 캐시 파일 제외
└── README.md                   # 프로젝트 개요 및 실행 가이드
```

---

## 🛠 설치 방법 (Installation)

가상환경(venv 또는 conda)을 활성화한 후 아래 명령어로 필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

---

## 🚀 실행 가이드 (Usage)

### 1. 설정(Config) 기반 모델 학습 (`train.py`)
`configs/default_config.yaml`에 정의된 하이퍼파라미터를 바탕으로 전체 파이프라인(데이터 로드 ➔ 열화 증강 ➔ Zero-shot 평가 ➔ LoRA 파인튜닝 ➔ 재평가)이 자동으로 실행됩니다.

```bash
# 기본 config(configs/default_config.yaml)를 적용하여 실행
python src/train.py

# 특정 config 파일을 지정하여 실행
python src/train.py --config configs/default_config.yaml

# CLI 인자를 전달하여 특정 하이퍼파라미터 오버라이드
python src/train.py --epochs 3 --batch_size 4 --learning_rate 0.0005 --output_dir checkpoints/exp_01
```

### 2. 학습 완료 모델 평가 (`evaluate.py`)
학습된 LoRA 체크포인트 또는 사전학습 모델에 대해 테스트 데이터셋에서 WER 및 CER 지표를 독립적으로 평가하고 결과를 저장할 수 있습니다.

```bash
# 기본 사전학습 모델의 전장 열화 환경 평가
python src/evaluate.py --model_id openai/whisper-small

# 학습된 LoRA 체크포인트 평가 및 결과 JSON 저장
python src/evaluate.py \
    --checkpoint_dir checkpoints/tactical_whisper \
    --metrics wer cer \
    --output_file checkpoints/eval_results.json
```

### 3. 단위 테스트 실행 (`pytest`)
작성된 오디오 전처리 및 텐서 연산 로직이 차원 에러나 비정상 수치(NaN, Inf) 없이 안전하게 동작하는지 검증합니다:

```bash
pytest tests/ -v
```

### 4. 인터랙티브 탐색 및 EDA (`notebooks/exploration.ipynb`)
주피터 랩 또는 VS Code에서 `notebooks/exploration.ipynb`를 열어 아래 작업들을 인터랙티브하게 확인하실 수 있습니다:
* 원본 Clean 오디오 청취 및 파형 시각화
* 구형 무전기 필터링 및 노이즈 합성 오디오 비교 (Waveform & Spectrogram)
* Whisper Zero-shot 예측 vs 정답 레이블 텍스트 비교
* LoRA 학습 파라미터 구조 확인
