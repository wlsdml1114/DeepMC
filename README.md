# DeepMC
### DeepMC architecture
* DeepMC(Kumar, Peeyush, et al. "Micro-climate Prediction-Multi Scale Encoder-decoder based Deep Learning Framework." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.) 논문의 모델을 구현하기 위해 만든 프로젝트
<img width="1428" alt="스크린샷 2022-05-23 오후 5 20 25" src="https://user-images.githubusercontent.com/6532977/173773577-b3dc8f2d-1332-4045-bf79-10d9ff7d4b5c.png">

## DeepMC 코드 구조
* Encoder
  * CNN-LSTM, CNN stacks
  * net/encoder.py
    * LSTMstack
    * CNNstack
* 2 level Attention layer
  * Position based content attention, Scaled guided attention
  * net/attention.py
    * Position_based_content_attention
    * Scaled_Guided_Attention
* Decoder
  * LSTM with FC-layer
  * net/decoder.py
    * Decoder
* All network
  * Encoder, 2 level attention, Decoder
  * net/deepmc.py
    * DeepMC

## Library version
* python==3.8.5
* pytorch==1.8.1
* torchvision==0.9.1
* opencv-python==4.5.2.54
* pytorch-lightning==1.3.8

## 실행 방법
```bash
python trainer.py
```