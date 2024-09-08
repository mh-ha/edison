# edison

**edison 폴더 구조**
- `__pycache__/`
- `configs/`
    - config 관련 폴더입니다.
- `constants/`
    - generation과 관련된 constant 파일을 가지고 있습니다.
- `layers/`
    - 주요 layer class를 저장한 폴더입니다.
- `metrics/`
    - 주요 metric을 계산하는 코드를 가지고 있습니다.
- `modules/`
    - lightning module을 가지고 있습니다.
- `pipes/`
    - modules의 클래스들을 동작하는 코드를 가지고 있습니다.
- `repositories/`
- `resources/`
- `schemas/`
    - diffusion model의 output을 정의한 클래스를 가지고 있습니다.
- `tests/`
- `utils/`
    - 주요 계산 코드들을 모아두었습니다.
 
진입점은 다음 두 파일 중 하나로 시작하시면 됩니다.

- [`run.py`](http://run.py): latent diffusion
- `run_discrete_diffusion.py` : discrete diffusion

shell script로 시작하실 수도 있습니다.

- `scripts/train.sh`

training, evaluation pipeline은 다음 파일들에서 보실 수 있습니다.

- `pipes/train.py` : training 전체 파이프라인
- `pipes/trainer.py` : trainer 설정
- `pipes/evaluate.py` : evaluation 전체 파이프라인

training의 세부 사항을 지정하려면 다음 파일들을 참고해 주십시오.

- `modules/lightning_modules.py` : 뉴럴 네트워크 정의 부분. `layers/` 의 모델들을 여기서 불러와 사용합니다.
- `modules/lightning_data_modules.py` : dataloader 정의 부분. LD4LG의 data 전처리 부분을 그대로 포팅해 왔음.

뉴럴 네트워크의 세부 정의는 아래 파일들을 보시면 됩니다.

- `layers/diffusion.py` : latent diffusion과 discrete diffusion 모두 정의되어 있습니다.
- `layers/autoencoder.py` : autoencoder가 정의되어 있습니다.
- `layers/lm.py` : LM(BART) 불러오는 부분이 정의되어 있습니다.

config는 아래 파일들을 참고해 주십시오.

- `configs/base.py` : latent diffusion 관련 config
- `configs/discrete_diffusion.py` : word diffusion 관련 config

LD4LG 코드는 아래에서 보실 수 있습니다.

- `legacy/source_LD4LG/`

MDLM 코드는 아래에서 보실 수 있습니다.

- `mdlm/`
