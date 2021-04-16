## CasRel
CasRel是2020年ACL上一篇论文，是关系抽取的新作，论文名称为A Novel Cascade Binary Tagging Framework for
Relational Triple Extraction。本代码是在中文数据集上实现的。
## 数据集
   ccks2019关系抽取的数据集。将数据集处理为下面的格式：
 ```
{
    "text": "如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈",
    "triple_list": [
      [
        "喜剧之王",
        "主演",
        "周星驰"
      ]
    ]
  }
```
## requirement
```
tqdm
codecs
keras-bert = 0.81.1
tensorflow-gpu = 1.13.1
```
## 代码结构
```
|__train.py
|__data_loader.py
|__model.py
|__parse.py
|__utils.py
|__get_pb_model.py
|__predict.py
|__get_tf_pb_model.py
|__app.py
```
## 模型结果
```
f1: 0.7827, precision: 0.7736, recall: 0.7921, best f1: 0.7944
```
## 预测结果
```
{
    "text": "《爱的魔幻秀》是安心亚演唱的歌曲，由吴易伟作词，MartinHansen/StefanDouglasHayOsson作曲，收录于专辑《单身极品》中",
    "triple_list_gold": [
        {
            "subject": "爱的魔幻秀",
            "relation": "所属专辑",
            "object": "单身极品"
        },
        {
            "subject": "爱的魔幻秀",
            "relation": "歌手",
            "object": "安心亚"
        }
    ],
    "triple_list_pred": [
        {
            "subject": "爱的魔幻秀",
            "relation": "所属专辑",
            "object": "单身极品"
        },
        {
            "subject": "爱的魔幻秀",
            "relation": "歌手",
            "object": "安心亚"
        },
        {
            "subject": "爱的魔幻秀",
            "relation": "作词",
            "object": "吴易伟"
        }
    ],
    "new": [
        {
            "subject": "爱的魔幻秀",
            "relation": "作词",
            "object": "吴易伟"
        }
    ],
    "lack": []
}
```

