# OCR-deepLearning

# OCR-deeplearning
| Họ và tên        | MSSV       |
| -------------    | ---------- |
| Nguyễn Đức Long  | 20183583   |
| Đoàn Ngọc Khánh  | 20180110   |
| Nguyễn Phi Phúc  | 20180155   |
| Dương Bá Tân     | 20183978   |

## Train model Recognize 
```
cd Text-Recognition
python train_seq2seq.py --name test 
```
### Parameter 
* ```stride_size``` List stride_size của AvgPool2D của model VGG
* ```kernel_size``` List kernel của AvgPool2D của model VGG
* ```emb_dim``` Chiều của text sau lớp embedding
* ```enc_hid_dim``` Chiều ẩn ở Encoder, chiều ra của LSTM
* ```dec_hid_dim``` Chiều của RNN ở Decoder
* ```epoch``` Số epoch
* ```dropout``` Hệ số dropout
* ``` lr``` Learning rate
* ```batch_size``` Batch size
* ```rnn_type``` Default: "lstm" Option: ['lstm','gru','rnn'] Kiểu RNN ở Encoder
* ```seed``` seed 
* ```name``` tên session trên wandb
