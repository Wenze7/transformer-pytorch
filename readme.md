### 1 数据处理

- 数据为中英数据集，使用subword-nmt工具包对数据进行初步分词处理

  - ```shell
    subword-nmt learn-joint-bpe-and-vocab -i news-commentary-v13.zh-en.en -o .\code_en.file --write-vocabulary voc_en.txt
    
    subword-nmt learn-joint-bpe-and-vocab -i news-commentary-v13.zh-en.zh -o .\code_zh.file --write-vocabulary voc_zh.txt
    ```

    这一步会生成词典

  - ```shell
    subword-nmt apply-bpe -i news-commentary-v13.zh-en.en -c .\code_en.file -o en.txt 
    
    subword-nmt apply-bpe -i news-commentary-v13.zh-en.zh -c .\code_zh.file -o zh.txt 
    ```

    这一步使用词典对原始数据进行分词

- 经过subword-nmt的初始处理后，需要对数据进行word->id映射，截断和pad，生成训练集和测试集等，具体对应的code文件为 **DataProcesser.py**

### 2 batch数据获取器

​	根据batch_size获取数据，具体对应的code文件为**BatchDataGenerater.py**

### 3 模型架构

模型的架构主要如下：

- Transformer
  - TransformerEncoder
    - token embedding + pos embedding
    - EncoderLayer * 6
      - SelfMultiHeadAttention
      - Add+Norm
      - FeedForward
      - Add+Norm
  - TransformerDecoder
    - token embedding + pos embedding
    - EncoderLayer * 6
      - SelfMultiHeadAttention
      - Add+Norm
      - CrossMultiHeadAttention
      - Add+Norm
      - FeedForward
      - Add+Norm
  - FC

具体对应的code文件为Models.py

### 4 Mask

- attention mask

  - 在encoder中对pad部分进行mask
  - 在decoder中对pad和后面的部分进行mask，为两个mask的或
  - 但是在inference阶段可以不对encoder进行pad的mask

  对应的code为**utils.py**中的**pad_mask()**和**subsequent_mask()**

- Loss mask

  - 对pad部分进行loss的mask

  对应的code为**utils.py**中的**MaskedSoftmaxCELoss**

### 5 训练

seq2seq模型的训练方式

### 6 推断

- beam search
  - 数据并行的beam search，以batch为单位进行，删除一个batch中已经生成的句子，减少运算量，并设置删除前和删除后index映射。并行的beam search可以大大提高推断的速度。对应**utils.py**中的**beam_search()**
  - 数据串行的beam search，一条一条的生成，速度太慢。具体对应**utils.py**中最后被注释的部分。
- bleu值的计算
  - 采用4-gram

### 7 参考文献

- [2.2.1-Pytorch编写Transformer.md (datawhalechina.github.io)](https://datawhalechina.github.io/learn-nlp-with-transformers/#/./篇章2-Transformer相关原理/2.2.1-Pytorch编写Transformer)
- [jadore801120/attention-is-all-you-need-pytorch at 132907dd272e2cc92e3c10e6c4e783a87ff8893d (github.com)](https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/132907dd272e2cc92e3c10e6c4e783a87ff8893d)







