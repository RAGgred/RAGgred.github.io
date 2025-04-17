---
title: "Logistic Regression, Random Forest and XGBoost performance comparison on a Bank Marketing Dataset"
date: 2025-04-15 14:11:00 +0100
categories: [machine learning]
tags: [statistical modelling, logistic regression, random forest, xgboost]
render_with_liquid: false
---

This project analyzes a Portuguese bank's telemarketing campaign dataset to predict whether a client will subscribe to a long-term deposit. Using real-world customer data and multiple statistical models—including Logistic Regression, Random Forest, and XGBoost—the project showcases data preprocessing, exploratory data analysis, and predictive modeling to optimize campaign strategies.



---


##  Full code 
 
 You can check out the full code used for this project [here.](https://github.com/RAGgred/RAGgred.github.io/blob/main/assets/projects/notebooks/statisticalmodelling.ipynb)
 
## 📁 Dataset

- 📌 Source: [UCI Machine Learning Repository – Bank Marketing Dataset](https://doi.org/10.24432/C5K306)  
- 📅 Period: 2008–2013  
- 🔢 Observations: 4,100  
- 🎯 Target: Subscription to term deposit (`y`)

---

## 🔧 Data pre-processing

The dataset is in very good condition with no missing values; however, some features (in job, education, housing, loan, default, marital and k columns) include the value ‘unknown’ to indicate missing data.  The dataset is also imbalanced, with fewer clients subscribing, making the AUC recall more important than accuracy when it comes to future model performance (Saito et all, 2015).

The data cleansing strategy chosen for the ‘unknown’ values was handling the unknown data as its own category over mode imputation. Although both strategies had the same accuracy, handling ‘unknown as its own category presents some advantages. It performs better on recall (catches more ‘yes’ responses: 47% vs 46%) and it gives a slight edge in macro average. Additionally, when performing three based models, the unknown can represent a real-life signal such as customers refusing calls, which helps avoiding assumptions in data. 

![Data pre-processing](https://RAGgred.github.io/assets/projects/images/missingdatastrat.png)

---

## 🔍  EDA

Exploratory analysis was undertaken to understand the structures and relationships between variables and support with model choosing and development. Distributions of numerical features such as age and duration were visualized using histograms with Kernel Density Estimation overlays, which are ideal for understanding the distribution of continuous variables. The histograms are useful to spot skewness, outliers and sparsity (Waksom, 2021). Overall, the distribution show skewed and nonlinear distributions which support the case for using models such as XGBoost and Random Forest which can handle non-linear features well (Wohlwend 2023).

The most notable feature in the distribution plots is the call length as shown below: 
 
![duration distribution](https://RAGgred.github.io/assets/projects/images/disofduration.png)

The right-skewness indicates that the majority of calls are short, with some extremely long one with longer calls being more likely to transform into conversions as shown in figure 3. This feature can serve as a strong proxy to gauge customer engagement; however, it may not be suitable for real-life predictions as the conversion is known only after the call ended.
 
![Call duration by subscription outcome](https://RAGgred.github.io/assets/projects/images/calldurationbyoutcome.png)

Another notable feature is pdays, which shows the number of days after the client was last contacted from a previous campaign. This feature is notable, as previous contact significantly influences the conversion rate. The plot is left-skewed and presents a significant spike at 999 which indicates that the majority of the clients have not been contacted yet. 
 
![Distribution of pday](https://RAGgred.github.io/assets/projects/images/disofpday.png)

When it comes to the current campaign, most clients were contacted once or twice. This is significant as multiple contacts can decrease conversion effectiveness due to customer fatigue. Capping the number of times a customer is contacted could be useful in avoiding customer fatigue (Wang et all, 2022).
 
![Distribution of campaign](https://RAGgred.github.io/assets/projects/images/disofcamp.png)

When it comes to external factors, the employment variation is most significant for conversion, where most values are concentrated in the positive range, indicating that stable or recovering periods are more likely to see customer conversions. 
 
![Distribution of emp.var.rate.](https://RAGgred.github.io/assets/projects/images/disofempvarrate.png)

 
The correlation heatmap shows strong positive correlations between four variables: employment variation rate, consumer confidence index, euribor 3-month rate and number of employees. This makes sense in the context of macroeconomics, where economic strength affects multiple indicators. 

There are also strong negative correlations between emp.var.rate and pdays indicating that more recent follow ups have fewer previous contacts, and emp.var.rate and previous, indicating that when the employment rate is higher less calls are needed to secure a conversion. 
 
![Corelation heatmap](https://RAGgred.github.io/assets/projects/images/corrheat.png)

There is weak or no corelation for age, campaign and duration suggesting that these fields contain independent information. These variables have been measured against the target variable ‘y’ and visualised using boxplots. Boxplots have been chosen as they are built to provide high-level information at a glance, including the IQR and outliers, and it makes it easy to make comparison between different groups (Yi, 2024).

When it comes to age, older individuals are slightly more inclined to subscribe. This can be due to potential greater income security. The duration is the strongest indicator with longer calls being significantly more likely to convert, whereas the fewer campaigns are more effective in securing conversion, as over contacting may cause fatigue. 


---

## 📈 Statistical Modelling and model comparision
The aim of this analysis is to assess whether a client will subscribe to a long-term deposit or not. The binary classification problem was solved by employing Logistic Regression, Random Forest and XGBoost.

| Model              | Accuracy | ROC AUC |
|--------------------|----------|---------|
| Logistic Regression | 92.4%    | 95%     |
| Random Forest       | ~91%     | ~94%    |
| XGBoost             | ~91%     | 95%+    |




![MP1](https://RAGgred.github.io/assets/projects/images/modelperfcomp.png)
A performance comparison among three classifiers showed that logistic regression achieved the highest accuracy at 92.4% and a ROC AUC of 95%, as illustrated in Figure 9. Bar charts are used in this analysis because they provide a clear way to display the distribution of data points and compare metric values across different categories (Yi, 2024). This finding suggests a strong alignment between the linear decision boundary and the relationship between the features and the target variable. Random forest and XGBoost performed similarly, although XGBoost exhibited a slightly higher ROC AUC, indicating a better ability to discriminate between classes. These results reinforce the use of logistic regression as a baseline model, with XGBoost serving as a suitable alternative for more complex tasks.




![MP2](https://RAGgred.github.io/assets/projects/images/modelperfcomp2.png)
The classification report and confusion matrix for the Linear regression model show that the model achieves high accuracy (92%) and strong precision for both classes. However, it struggles with identifying true positives: only 45% of clients who subscribed were correctly predicted as such. This highlights a trade-off between overall performance and sensitivity to the positive class. In the context of marketing, improving recall for the "yes" class could lead to better campaign targeting and conversion rates. 





![MP3](https://RAGgred.github.io/assets/projects/images/roc.png)
The ROC curve for the Logistic Regression model shows strong performance, with an AUC score of 0.94. This indicates that the model is highly effective at distinguishing between the positive class (yes) and the negative class (no). The curve is close to the top-left corner, which suggests that the model achieves a high true positive rate while minimizing false positives. In cases of imbalanced classification, the ROC-AUC score is a more reliable metric than accuracy. This further supports the use of Logistic Regression as a strong baseline model for this task.  



---

## ✅ Conclusion & Recommendations

- Use **Logistic Regression** as a baseline for campaign prediction.
- Consider economic indicators and call length in strategic planning.
- Future improvements: Use **cost-sensitive learning** or **SMOTE** for better recall on minority class.

---

## 📚 References

Key sources include peer-reviewed papers, technical blogs (e.g., Atlassian, Medium, Forbes), and project-specific research:

- Anderson, C. (2019). Hot or Not? Heatmaps and Correlation Matrix Plots. [online] Medium. Available at: https://medium.com/@connor.anderson_42477/hot-or-not-heatmaps-and-correlation-matrix-plots-940088fa2806.
- Moro, S., Cortez, P. and Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, pp.22–31.
- Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. 
- Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), p.e0118432. doi:https://doi.org/10.1371/journal.pone.0118432.
- Tékouabou, S.C.K., Gherghina, Ş.C., Toulni, H., Neves Mata, P., Mata, M.N. and Martins, J.M. (2022). A Machine Learning Framework towards Bank Telemarketing Prediction. Journal of Risk and Financial Management, 15(6), p.269. doi:https://doi.org/10.3390/jrfm15060269.
- Ugenti, M. (2024). Council Post: The Evolution Of Direct Marketing. Forbes. [online] 12 Aug. Available at: https://www.forbes.com/councils/forbescommunicationscouncil/2023/04/14/the-evolution-of-direct-marketing/.
- Wang, C.X., Yuan, H. and Beck, J.T. (2022). Too tired for a good deal: How customer fatigue shapes the performance of Pay-What-You-Want pricing. Journal of Business Research, 144, pp.987–996. doi:https://doi.org/10.1016/j.jbusres.2022.02.014.
- Waskom, M. (2021). Seaborn: Statistical Data Visualization. Journal of Open Source Software, [online] 6(60), p.3021. doi:https://doi.org/10.21105/joss.03021.
- Wohlwend, B. (2023). Decision Tree, Random Forest, and XGBoost: An Exploration into the Heart of Machine Learning. [online] Medium. Available at: https://medium.com/@brandon93.w/decision-tree-random-forest-and-xgboost-an-exploration-into-the-heart-of-machine-learning-90dc212f4948.
- Xie, C., Zhang, J.-L., Zhu, Y., Xiong, B. and Wang, G.-J. (2023). How to improve the success of bank telemarketing? Prediction and interpretability analysis based on machine learning. Computers & Industrial Engineering, 175, p.108874. doi:https://doi.org/10.1016/j.cie.2022.108874.
- Yi, M. (2024a). A Complete Guide to Bar Charts. [online] Atlassian. Available at: https://www.atlassian.com/data/charts/bar-chart-complete-guide.
- Yi, M. (2024b). A Complete Guide to Box Plots. [online] Atlassian. Available at: https://www.atlassian.com/data/charts/box-plot-complete-guide.


