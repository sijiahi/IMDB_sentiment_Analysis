# IMDB_sentiment_Analysis
## 模型训练细节
项目最终共展示三个模型，三个模型的训练细节可分别在：
   * Tfidf_train.py(model1): 进行预处理后进行 tfidf 向量化训练（特征维度为
10000）；
   * Tfidf_train_feat3000.py(model3): 进 行预 处 理 后进行 tfidf 向 量 化 训 练 （特
征维度为 3000）；
   * Bow_train.py(model2): 进 行 预 处 理 后 使 用 词 袋 模 型 进 行 向 量 化 后 训 练 （ 特
征维度为 10000）；
## 模型训练细节
### 模型性能测试
可对三个模型性能进行测试，测试文件为 statostical_analyze.py，文件中默认
对 model1 进行测试，若希望测试其他模型，可在主函数中通过修改测试模型来进
行。
