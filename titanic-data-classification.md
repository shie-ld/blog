
# Machine Learning with Python: Classification (complete tutorial)

Machine Learning with Python: Classification (complete tutorial)

### Data Analysis & Visualization, Feature Engineering & Selection, Model Design & Testing, Evaluation & Explainability

### Summary

In this article, using Data Science and Python, I will explain the main steps of a Classification use case, from data analysis to understanding the model output.

![](https://cdn-images-1.medium.com/max/2000/1*gS5PdcX1sk1yQgYQnRSGcg.png)

Since this tutorial can be a good starting point for beginners, I will use the “**Titanic dataset**” from the famous Kaggle competition, in which you are provided with passengers data and the task is to build a predictive model that answers the question: “what sorts of people were more likely to survive?” (linked below).
[**Titanic: Machine Learning from Disaster**
*Start here! Predict survival on the Titanic and get familiar with ML basics*www.kaggle.com](https://www.kaggle.com/c/titanic/overview)

I will present some useful Python code that can be easily used in other similar cases (just copy, paste, run) and walk through every line of code with comments, so that you can easily replicate this example. 
In particular, I will go through:

* Environment setup: import libraries and read data

* Data Analysis: understand the meaning and the predictive power of the variables

* Feature engineering: extract features from raw data

* Preprocessing: data partitioning, handle missing values, encode categorical variables, scale

* Feature Selection: keep only the most relevant variables

* Model design: train, tune hyperparameters, validation, test

* Performance evaluation: read the metrics

* Explainability: understand how the model produces results

### Setup

First of all, I need to import the following libraries.

    **## for data**
    import **pandas **as pd
    import **numpy **as np

    **## for plotting**
    import **matplotlib**.pyplot as plt
    import **seaborn **as sns

    **## for statistical tests**
    import **scipy**
    import **statsmodels**.formula.api as smf
    import statsmodels.api as sm

    **## for machine learning**
    from **sklearn **import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

    **## for explainer**
    from **lime **import lime_tabular

Then I will read the data into a pandas Dataframe.

    dtf = pd.read_csv('data_titanic.csv')
    dtf.head()

![](https://cdn-images-1.medium.com/max/2608/1*ur-xUqrNt4qoHV8NnLUeXQ.png)

Details about the columns can be found in the provided link to the dataset.

Please note that each row of the table represents a specific passenger (or observation). If you are working with a different dataset that doesn’t have a structure like that, in which each row represents an observation, then you need to summarize data and transform it.

Now that it’s all set, I will start by analyzing data, then select the features, build a machine learning model and predict.

Let’s get started, shall we?

### Data Analysis

In statistics, [exploratory data analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) is the process of summarizing the main characteristics of a dataset to understand what the data can tell us beyond the formal modeling or hypothesis testing task.

I always start by getting an overview of the whole dataset, in particular I want to know how many **categorical** and **numerical** variables there are and the proportion of **missing data**. Recognizing a variable’s type sometimes can be tricky because categories can be expressed as numbers (the Su*rvived c*olumn is made of 1s and 0s). To this end, I am going to write a simple function that will do that for us:

    **'''
    Recognize whether a column is numerical or categorical.
    :parameter
        :param dtf: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of unique values to recognize a column as categorical
    :return
        "cat" if the column is categorical or "num" otherwise
    '''**
    def **utils_recognize_type**(dtf, col, max_cat=20):
        if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
            return **"cat"**
        else:
            return **"num"**

This function is very useful and can be used in several occasions. To give an illustration I’ll plot a [**heatmap](http://Heat map) **of the dataframe to visualize columns type and missing data.

    dic_cols = {col:**utils_recognize_type**(dtf, col, max_cat=20) for col in dtf.columns}

    heatmap = dtf.isnull()
    for k,v in dic_cols.items():
     if v == "num":
       heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
     else:
       heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)

    sns.**heatmap**(heatmap, cbar=False).set_title('Dataset Overview')
    plt.show()

    print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")

![](https://cdn-images-1.medium.com/max/2418/1*YNeaA2mB5kuXkn80Yb8vpQ.png)

There are 885 rows and 12 columns:

* each row of the table represents a specific passenger (or observation) identified by *PassengerId*, so I’ll set it as index (or [primary key](https://en.wikipedia.org/wiki/Primary_key) of the table for SQL lovers).

* *Survived* is the phenomenon that we want to understand and predict (or target variable), so I’ll rename the column as “*Y”*. It contains two classes: 1 if the passenger survived and 0 otherwise, therefore this use case is a binary classification problem.

* *Age *and *Fare *are numerical variables while the others are categorical.

* Only *Age *and *Cabin *contain missing data.

    dtf = dtf.set_index("**PassengerId**")

    dtf = dtf.rename(columns={"**Survived**":"**Y**"})

I believe visualization is the best tool for data analysis, but you need to know what kind of plots are more suitable for the different types of variables. Therefore, I’ll provide the code to plot the appropriate visualization for different examples.

First, let’s have a look at the univariate distributions (probability distribution of just one variable). A [**bar plot](https://en.wikipedia.org/wiki/Bar_chart) **is appropriate to understand labels frequency for a single **categorical **variable. For example, let’s plot the target variable:

    **y = "Y"**

    ax = dtf[y].value_counts().sort_values().plot(kind="barh")
    totals= []
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax.patches:
         ax.text(i.get_width()+.3, i.get_y()+.20, 
         str(round((i.get_width()/total)*100, 2))+'%', 
         fontsize=10, color='black')
    ax.grid(axis="x")
    plt.suptitle(y, fontsize=20)
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*-KhpzEZfVUFdFfBnf54xIw.png)

Up to 300 passengers survived and about 550 didn’t, in other words the survival rate (or the population mean) is 38%.

Moreover, a [**histogram](https://en.wikipedia.org/wiki/Histogram)** is perfect to give a rough sense of the density of the underlying distribution of a single **numerical **data. I recommend using a [**box plot](https://en.wikipedia.org/wiki/Box_plot) **to graphically depict data groups through their quartiles. Let’s take the *Age *variable for instance:

    **x = "Age"**

    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    fig.suptitle(x, fontsize=20)

    **### distribution**
    ax[0].title.set_text('distribution')
    variable = dtf[x].fillna(dtf[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable < 
                        breaks[10]) ]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
    des = dtf[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    **### boxplot **
    ax[1].title.set_text('outliers (log scale)')
    tmp_dtf = pd.DataFrame(dtf[x])
    tmp_dtf[x] = np.log(tmp_dtf[x])
    tmp_dtf.boxplot(column=x, ax=ax[1])
    plt.show()

![](https://cdn-images-1.medium.com/max/2564/1*PhXWmvd8Weg3cmWNsVO2XQ.png)

The passengers were, on average, pretty young: the distribution is skewed towards the left side (the mean is 30 y.o and the 75th percentile is 38 y.o.). Coupled with the outliers in the box plot, the first spike in the left tail says that there was a significant amount of children.

I’ll take the analysis to the next level and look into the bivariate distribution to understand if *Age* has predictive power to predict *Y*. This would be the case of **categorical (*Y*) vs numerical (*Age*)**, therefore I shall proceed like this:

* split the population (the whole set of observations) into 2 samples: the portion of passengers with *Y = 1* (Survived) and *Y = 0 *(Not Survived).

* Plot and compare densities of the two samples, if the distributions are different then the variable is predictive because the two groups have different patterns.

* Group the numerical variable (*Age*) in bins (subsamples) and plot the composition of each bin, if the proportion of 1s is similar in all of them then the variable is not predictive.

* Plot and compare the box plots of the two samples to spot different behaviors of the outliers.

    **cat, num = "Y", "Age"**

    fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False)
    fig.suptitle(x+"   vs   "+y, fontsize=20)
                
    **### distribution**
    ax[0].title.set_text('density')
    for i in dtf[cat].unique():
        sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
    ax[0].grid(True)

    **### stacked**
    ax[1].title.set_text('bins')
    breaks = np.quantile(dtf[num], q=np.linspace(0,1,11))
    tmp = dtf.groupby([cat, pd.cut(dtf[num], breaks, duplicates='drop')]).size().unstack().T
    tmp = tmp[dtf[cat].unique()]
    tmp["tot"] = tmp.sum(axis=1)
    for col in tmp.drop("tot", axis=1).columns:
         tmp[col] = tmp[col] / tmp["tot"]
    tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)

    **### boxplot **  
    ax[2].title.set_text('outliers')
    sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2])
    ax[2].grid(True)
    plt.show()

![](https://cdn-images-1.medium.com/max/3594/1*ZhI9R4JiKE_kPDpeVez6YA.png)

These 3 plots are just different perspectives of the conclusion that *Age *is predictive. The survival rate is higher for younger passengers: there is a spike in the left tail of 1s distribution and the first bin (0–16 y.o.) contains the highest percentage of survived passengers.

When not convinced by the “eye intuition”, you can always resort to good old statistics and run a test. In this case of categorical (*Y*) vs numerical (*Age*), I would use a **o[ne-way ANOVA test](http://en.wikipedia.org/wiki/F_test#One-way_ANOVA_example)**. Basically, it tests whether the means of two or more independent samples are significantly different, so if the p-value is small enough (<0.05) the null hypothesis of samples means equality can be rejected.

    **cat, num = "Y", "Age"**

    model = smf.**ols**(num+' ~ '+cat, data=dtf).fit()
    table = sm.stats.**anova_lm**(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")

![](https://cdn-images-1.medium.com/max/2000/1*OA5Micu7fM-OhtnPOdne-A.png)

Apparently the passengers' age contributed to determine their survival. That makes sense as the lives of women and children were to be saved first in a life-threatening situation, typically abandoning ship, when survival resources such as lifeboats were limited (the “[women and children first](https://en.wikipedia.org/wiki/Women_and_children_first)” code).

In order to check the validity of this first conclusion, I will have to analyze the behavior of the *Sex *variable with respect to the target variable. This is a case of **categorical (*Y*) vs categorical (*Sex*)**, so I’ll plot 2 bar plots, one with the amount of 1s and 0s among the two categories of *Sex *(male and female) and the other with the percentages.

    **x, y = "Sex", "Y"**

    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    fig.suptitle(x+"   vs   "+y, fontsize=20)

    **### count**
    ax[0].title.set_text('count')
    order = dtf.groupby(x)[y].count().index.tolist()
    sns.catplot(x=x, hue=y, data=dtf, kind='count', order=order, ax=ax[0])
    ax[0].grid(True)

    **### percentage**
    ax[1].title.set_text('percentage')
    a = dtf.groupby(x)[y].count().reset_index()
    a = a.rename(columns={y:"tot"})
    b = dtf.groupby([x,y])[y].count()
    b = b.rename(columns={y:0}).reset_index()
    b = b.merge(a, how="left")
    b["%"] = b[0] / b["tot"] *100
    sns.barplot(x=x, y="%", hue=y, data=b,
                ax=ax[1]).get_legend().remove()
    ax[1].grid(True)
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*C2rzIv-IjCI6p5EsSEKaNQ.png)

More than 200 female passengers (75% of the total amount of women onboard) and about 100 male passengers (less than 20%) survived. To put it another way, among women the survival rate is 75% and among men is 20%, therefore *Sex *is predictive. Moreover, this confirms that they gave priority to women and children.

Just like before, we can test the correlation of these 2 variables. Since they are both categorical, I’d use a C[**hi-Square test:](https://en.wikipedia.org/wiki/Chi-square_test)** assuming that two variables are independent (null hypothesis), it tests whether the values of the contingency table for these variables are uniformly distributed. If the p-value is small enough (<0.05), the null hypothesis can be rejected and we can say that the two variables are probably dependent. It’s possible to calculate C[**ramer’s V ](https://en.wikipedia.org/wiki/Cram%C3%A9r's_V)t**hat is a measure of correlation that follows from this test, which is symmetrical (like traditional Pearson’s correlation) and ranges between 0 and 1 (unlike traditional Pearson’s correlation there are no negative values).

    **x, y = "Sex", "Y"**

    cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
    chi2_test = scipy.stats.**chi2_contingency**(cont_table)
    chi2, p = chi2_test[0], chi2_test[1]
    n = cont_table.sum().sum()
    phi2 = chi2/n
    r,k = cont_table.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
    coeff, p = round(coeff, 3), round(p, 3)
    conclusion = "Significant" if p < 0.05 else "Non-Significant"
    print("Cramer Correlation:", coeff, conclusion, "(p-value:"+str(p)+")")

![](https://cdn-images-1.medium.com/max/2000/1*KS4bwcy6pFpgXreDIwk-RA.png)

*Age *and *Sex* are examples of predictive features, but not all of the columns in the dataset are like that. For instance, *Cabin *seems to be a **useless variable** as it doesn’t provide any useful information, there are too many missing values and categories.

This kind of analysis should be carried on for each variable in the dataset to decide what should be kept as a potential feature and what can be dropped because not predictive (check out the link to the full code).

### Feature Engineering

It’s time to create new features from raw data using domain knowledge. I will provide one example: I’ll try to create a useful feature by extracting information from the *Cabin *column. I’m assuming that the letter at the beginning of each cabin number (i.e. “***B**96*”) indicates some kind of section, maybe there were some lucky sections near to lifeboats. I will summarize the observations in clusters by extracting the section of each cabin:

    **## Create new column**
    dtf["**Cabin_section**"] = dtf["**Cabin**"].apply(lambda x: str(x)[0])

    **## Plot contingency table
    **cont_table = pd.crosstab(index=dtf["**Cabin_section"**], 
                 columns=dtf["**Pclass**"], values=dtf["**Y**"], aggfunc="sum")

    sns.**heatmap**(cont_table, annot=True, cmap="YlGnBu", fmt='.0f',
                linewidths=.5).set_title( 
                'Cabin_section vs Pclass (filter: Y)' )

![](https://cdn-images-1.medium.com/max/2252/1*r9Pw1wtZ9DQunc76InAwmA.png)

This plot shows how survivors are distributed among cabin sections and classes (7 survivors are in section A, 35 in B, …). Most of the sections are assigned to the 1st and the 2nd classes, while the majority of missing sections (“*n”*) belongs to the 3rd class. I am going to keep this new feature instead of the column *Cabin:*

![](https://cdn-images-1.medium.com/max/2218/1*Wn2vfFKnu7yqsWkLmZLlcA.png)

### Preprocessing

Data preprocessing is the phase of preparing the raw data to make it suitable for a machine learning model. In particular:

1. each observation must be represented by a single row, in other words you can’t have two rows describing the same passenger because they will be processed separately by the model (the dataset is already in such form, so ✅). Moreover, each column should be a feature, so you shouldn’t use *PassengerId *as a predictor, that’s why this kind of table is called “**feature matrix**”.

1. The dataset must be **partitioned **into at least two sets: the model shall be trained on a significant portion of your dataset (so-called “train set”) and tested on a smaller set (“test set”).

1. **Missing values** should be replaced with something, otherwise your model may freak out.

1. **Categorical data** must be encoded, which means converting labels into integers, because machine learning expects numbers not strings.

1. It’s good practice to **scale **the data, it helps to normalize the data within a particular range and speed up the calculations in an algorithm.

Alright, let’s begin by **partitioning the dataset**. When splitting data into train and test sets you must follow 1 basic rule: rows in the train set shouldn’t appear in the test set as well. That’s because the model sees the target values during training and uses it to understand the phenomenon. In other words, the model already knows the right answer for the training observations and testing it on those would be like cheating. I’ve seen a lot of people pitching their machine learning models claiming 99.99% of accuracy that did in fact ignore this rule. Luckily, the S*cikit-learn* package knows that:

    **## split data**
    dtf_train, dtf_test = **model_selection**.**train_test_split**(dtf, 
                          test_size=0.3)

    **## print info**
    print("X_train shape:", dtf_train.drop("Y",axis=1).shape, "| X_test shape:", dtf_test.drop("Y",axis=1).shape)
    print("y_train mean:", round(np.mean(dtf_train["Y"]),2), "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
    print(dtf_train.shape[1], "features:", dtf_train.drop("Y",axis=1).columns.to_list())

![](https://cdn-images-1.medium.com/max/2292/1*5RUyGcVjtyP0Z67rEsLK1g.png)

Next step: the *Age *column contains some **missing data** (19%) that need to be handled. In practice, you can replace missing data with a specific value, like 9999, that keeps trace of the missing information but changes the variable distribution. Alternatively, you can use the average of the column, like I’m going to do. I’d like to underline that from a Machine Learning perspective, it’s correct to first split into train and test and then replace *NAs* with the average of the training set only.

    dtf_train["Age"] = dtf_train["Age"].**fillna**(dtf_train["Age"].**mean**())

There are still some **categorical data** that should be encoded. The two most common encoders are the Label-Encoder (each unique label is mapped to an integer) and the One-Hot-Encoder (each label is mapped to a binary vector). The first one is suited for data with ordinality only. If applied to a column with no ordinality, like *Sex*, it would turn the vector *[m*ale, female, female, male, …] into [1, 2, 2, 1, …] and we would have that female > male and with an average of 1.5 which makes no sense. On the other hand, the One-Hot-Encoder would transform the previous example into two [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)) (dichotomous quantitative variables): Mal*e [1*, 0, 0, 1, …] and Fem*ale [0*, 1, 1, 0, …]. It has the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space, but features with high cardinality can lead to a dimensionality issue. I shall use the One-Hot-Encoding method, transforming 1 categorical column with n unique values into n-1 dummies. Let’s encode *Sex *as* *an example:

    **## create dummy**
    dummy = pd.get_dummies(dtf_train["**Sex**"], 
                           prefix="Sex",drop_first=True)
    dtf_train= pd.concat([dtf_train, dummy], axis=1)
    print( dtf_train.filter(like="Sex", axis=1).head() )

    **## drop the original categorical column**
    dtf = dtf_train.drop("**Sex**", axis=1)

![](https://cdn-images-1.medium.com/max/2000/1*OF-cFsJWSVw5tA8DeMCiDQ.png)

Last but not least, I’m going to **scale the features**. There are several different ways to do that, I’ll present just the most used ones: the Standard-Scaler and the MinMax-Scaler. The first one assumes data is normally distributed and rescales it such that the distribution centres around 0 with a standard deviation of 1. However, the outliers have an influence when computing the empirical mean and standard deviation which shrink the range of the feature values, therefore this scaler can’t guarantee balanced feature scales in the presence of outliers. On the other hand, the MinMax-Scaler rescales the data set such that all feature values are in the same range (0–1). It is less affected by outliers but compresses all inliers in a narrow range. Since my data is not normally distributed, I’ll go with the MinMax-Scaler:

    scaler = **preprocessing**.**MinMaxScaler**(feature_range=(0,1))
    X = scaler.fit_transform(dtf_train.drop("Y", axis=1))

    dtf_scaled= pd.DataFrame(X, columns=dtf_train.drop("Y", axis=1).columns, index=dtf_train.index)
    dtf_scaled["Y"] = dtf_train["Y"]
    dtf_scaled.head()

![](https://cdn-images-1.medium.com/max/2342/1*k0r2CHiCk9A16p-tfyefSA.png)

### Feature Selection

Feature selection is the process of selecting a subset of relevant variables to build the machine learning model. It makes the model easier to interpret and reduces overfitting (when the model adapts too much to the training data and performs badly outside the train set).

I already did a first “manual” feature selection during data analysis by excluding irrelevant columns. Now it’s going to be a bit different because we assume that all the features in the matrix are relevant and we want to drop the unnecessary ones. When a feature is not necessary? Well, the answer is easy: when there is a better equivalent, or one that does the same job but better.

I’ll explain with an example: *Pclass *is highly correlated with *Cabin_section* because, as we’ve seen before, certain sections were located in 1st class and others in the 2nd. Let’s compute the correlation matrix to see it:

    corr_matrix = dtf.copy()
    for col in corr_matrix.columns:
        if corr_matrix[col].dtype == "O":
             corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]

    corr_matrix = corr_matrix.**corr**(method="pearson")
    sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("pearson correlation")

![](https://cdn-images-1.medium.com/max/2648/1*ASqGwKzh9LfKyEHpwi82OQ.png)

One among *Pclass *and *Cabin_section *could be* *unnecessary and we may decide to drop it and keep the most useful one (i.e. the one with the lowest p-value or the one that most reduces entropy).

I will show two different ways to perform automatic feature selection: first I will use a regularization method** **and compare it with the ANOVA test already mentioned before, then I will show how to get feature importance from ensemble methods.

[**LASSO regularization](https://en.wikipedia.org/wiki/Lasso_(statistics))** is a regression analysis method that performs both variable selection and regularization in order to enhance accuracy and interpretability.

    X = dtf_train.drop("Y", axis=1).values
    y = dtf_train["Y"].values
    feature_names = dtf_train.drop("Y", axis=1).columns

    **## Anova**
    selector = **feature_selection.SelectKBest**(score_func=  
                   feature_selection.f_classif, k=10).fit(X,y)
    anova_selected_features = feature_names[selector.get_support()]
    
    **## Lasso regularization**
    selector = **feature_selection.SelectFromModel**(estimator= 
                  linear_model.LogisticRegression(C=1, penalty="l1", 
                  solver='liblinear'), max_features=10).fit(X,y)
    lasso_selected_features = feature_names[selector.get_support()]
     
    **## Plot
    **dtf_features = pd.DataFrame({"features":feature_names})
    dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
    dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
    dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
    dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
    dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
    dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
    sns.**barplot**(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)

![](https://cdn-images-1.medium.com/max/2206/1*JbLl3-xAk70ChrwOYf05_Q.png)

The blue features are the ones selected by both ANOVA and LASSO, the others are selected by just one of the two methods.

[**Random forest](https://en.wikipedia.org/wiki/Random_forest)** is an ensemble method that consists of a number of decision trees in which every node is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. Features importance is computed from how much each feature decreases the entropy in a tree.

    X = dtf_train.drop("Y", axis=1).values
    y = dtf_train["Y"].values
    feature_names = dtf_train.drop("Y", axis=1).columns.tolist()

    **## Importance**
    model = ensemble.**RandomForestClassifier**(n_estimators=100,
                          criterion="entropy", random_state=0)
    model.fit(X,y)
    importances = model.**feature_importances_**

    **## Put in a pandas dtf**
    dtf_importances = pd.DataFrame({"IMPORTANCE":importances, 
                "VARIABLE":feature_names}).sort_values("IMPORTANCE", 
                ascending=False)
    dtf_importances['cumsum'] =  
                dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")
        
    **##** **Plot**
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
        dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
                    kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, 
                                     legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), 
              xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()

![](https://cdn-images-1.medium.com/max/2660/1*Le11nI9ztW4j33LqwGecNQ.png)

It’s really interesting that *Age *and *Fare, *which are the most important features this time, weren’t the top features before and that on the contrary *Cabin_section E*, *F *and *D *don’t appear really useful here.

Personally, I always try to use less features as possible, so here I select the following ones and proceed with the design, train, test and evaluation of the machine learning model:

    X_names = ["Age", "Fare", "Sex_male", "SibSp", "Pclass_3", "Parch",
    "Cabin_section_n", "Embarked_S", "Pclass_2", "Cabin_section_F", "Cabin_section_E", "Cabin_section_D"]

    X_train = dtf_train[X_names].values
    y_train = dtf_train["Y"].values

    X_test = dtf_test[X_names].values
    y_test = dtf_test["Y"].values

Please note that before using test data for prediction you have to preprocess it just like we did for the train data.

### Model Design

Finally, it’s time to build the machine learning model. First, we need to choose an algorithm that is able to learn from training data how to recognize the two classes of the target variable by minimizing some error function.

![source: [scikit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)](https://cdn-images-1.medium.com/max/4156/1*rlwt7lMKqkgth0mhGK0LIw.png)*source: [scikit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)*

I suggest to always try a [**gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)** algorithm (like XGBoost). It’s a machine learning technique that produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. Basically it’s similar to a Random Forest with the difference that every tree is fitted on the error of the previous one.

![source: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)](https://cdn-images-1.medium.com/max/3862/1*Quo8-G6HK9KnyI8fHjhJSg.png)*source: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)*

There a lot of hyperparameters and there is no general rule about what is best, so you just have to find the right combination that fits your data better. You could do different tries manually or you can let the computer do this tedious job with a GridSearch (tries every possible combination but takes time) or with a RandomSearch (tries randomly a fixed number of iterations). I’ll try a RandonSearch for my **hyperparameter tuning**: the machine will iterate n times (1000) through training data to find the combination of parameters (specified in the code below) that maximizes a scoring function used as KPI (accuracy, the ratio of the number of correct predictions to the total number of input samples):

    **## call model**
    model = ensemble.**GradientBoostingClassifier**()

    **## define hyperparameters combinations to try
    **param_dic = {'**learning_rate**':[0.15,0.1,0.05,0.01,0.005,0.001],      *#weighting factor for the corrections by new trees when added to the model
    *'**n_estimators**':[100,250,500,750,1000,1250,1500,1750],  *#number of trees added to the model*
    '**max_depth**':[2,3,4,5,6,7],    *#maximum depth of the tree*
    '**min_samples_split**':[2,4,6,8,10,20,40,60,100],    *#sets the minimum number of samples to split*
    '**min_samples_leaf**':[1,3,5,7,9],     *#the minimum number of samples to form a leaf
    *'**max_features**':[2,3,4,5,6,7],     *#square root of features is usually a good starting point*
    '**subsample**':[0.7,0.75,0.8,0.85,0.9,0.95,1]}       *#the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.*

    **## random search**
    random_search = model_selection.**RandomizedSearchCV**(model, 
           param_distributions=param_dic, n_iter=1000, 
           scoring="accuracy").fit(X_train, y_train)

    print("Best Model parameters:", random_search.best_params_)
    print("Best Model mean accuracy:", random_search.best_score_)

    model = random_search.best_estimator_

![](https://cdn-images-1.medium.com/max/2664/1*Unw_2tw2VtqwxDMQmmbPiA.png)

Cool, that’s the best model, with a mean accuracy of 0.85, so probably 85% of predictions on the test set will be correct.

We can also validate this model using a **k-fold cross-validation**, a procedure that consists in splitting the data k times into train and validation sets and for each split the model is trained and tested. It’s used to check how well the model is able to get trained by some data and predict unseen data.

I’d like to clarify that I call **validation set **a set of examples used to tune the hyperparameters of a classifier, extracted from splitting training data. On the other end, a **test set** is a simulation of how the model would perform in production when it’s asked to predict observations never seen before.

It’s common to plot a **ROC curve **for every fold, a plot that illustrates how the ability of a binary classifier changes as its discrimination threshold is varied. It is created by plotting the true positive rate (1s predicted correctly) against the false positive rate (1s predicted that are actually 0s) at various threshold settings. The [**AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) **(area under the ROC curve) indicates the probability that the classifier will rank a randomly chosen positive observation (*Y=1*) higher than a randomly chosen negative one (*Y=0*).

Now I’ll show an example of with 10 folds (k=10):

    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0,1,100)
    fig = plt.figure()

    i = 1
    for train, test in cv.split(X_train, y_train):
       prediction = model.fit(X_train[train],
                    y_train[train]).predict_proba(X_train[test])
       fpr, tpr, t = metrics.roc_curve(y_train[test], prediction[:, 1])
       tprs.append(scipy.interp(mean_fpr, fpr, tpr))
       roc_auc = metrics.auc(fpr, tpr)
       aucs.append(roc_auc)
       plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = 
                %0.2f)' % (i, roc_auc))
       i = i+1
       
    plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = 
             %0.2f )' % (mean_auc), lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('K-Fold Validation')
    plt.legend(loc="lower right")
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*hMM9fION4eYgN6CyHKE00g.png)

According to this validation, we should expect an AUC score around 0.84 when making predictions on the test.

For the purpose of this tutorial I’d say that the performance is fine and we can proceed with the model selected by the RandomSearch. Once that the right model is selected, it can be trained on the whole train set and then tested on the test set.

    **## train**
    model.**fit**(X_train, y_train)

    **## test**
    predicted_prob = model.**predict_proba**(X_test)[:,1]
    predicted = model.**predict**(X_test)

In the code above I made two kinds of predictions: the first one is the probability that an observation is a 1, and the second is the prediction of the label (1 or 0). To get the latter you have to decide a probability threshold for which an observation can be considered as 1, I used the default threshold of 0.5.

### Evaluation

Moment of truth, we’re about to see if all this hard work is worth. The whole point is to study how many correct predictions and error types the model makes.

I’ll evaluate the model using the following common metrics: Accuracy, AUC, [**Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)**. I already mentioned the first two, but I reckon that the others are way more important. Precision is the fraction of 1s (or 0s) that the model predicted correctly among all predicted 1s (or 0s), so it can be seen as a sort of confidence level when predicting a 1 (or a 0). Recall is the portion of 1s (or 0s) that the model predicted correctly among all 1s (or 0s) in the test set, basically it’s the true 1 rate. Combining Precision and Recall with an armonic mean, you get the F1-score.

Let’s see how the model did on the test set:

    **## Accuray e AUC**
    accuracy = metrics.**accuracy_score**(y_test, predicted)
    auc = metrics.**roc_auc_score**(y_test, predicted_prob)
    print("Accuracy (overall correct predictions):",  round(accuracy,2))
    print("Auc:", round(auc,2))
        
    **## Precision e Recall**
    recall = metrics.**recall_score**(y_test, predicted)
    precision = metrics.**precision_score**(y_test, predicted)
    print("Recall (all 1s predicted right):", round(recall,2))
    print("Precision (confidence when predicting a 1):", round(precision,2))
    print("Detail:")
    print(metrics.**classification_report**(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

![](https://cdn-images-1.medium.com/max/2000/1*_W0WnLbAmeKYUBTMWBe1Ig.png)

As expected, the general accuracy of the model is around 85%. It predicted 71% of 1s correctly with a precision of 84% and 92% of 0s with a precision of 85%. In order to understand these metrics better, I’ll break down the results in a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix):

    classes = np.unique(y_test)
    fig, ax = plt.subplots()
    cm = metrics.**confusion_matrix**(y_test, predicted, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*9G51Re_0gPV5444sUeK1ng.png)

We can see that the model predicted 85 (70+15) 1s of which 70 are true positives and 15 are false positives, so it has a Precision of 70/85 = 0.82 when predicting 1s. On the other hand, the model got 70 1s right of all the 96 (70+26) 1s in the test set, so its Recall is 70/96 = 0.73.

Choosing a threshold of 0.5 to decide whether a prediction is a 1 or 0 led to this result. Would it be different with another one? Definitely yes, but there is no threshold that would bring the top score on both precision and recall, choosing a threshold means to make a compromise between these two metrics. I’ll show what I mean by plotting the ROC curve and the precision-recall curve of the test result:

    classes = np.unique(y_test)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    **## plot ROC curve**
    fpr, tpr, thresholds = metrics.**roc_curve**(y_test, predicted_prob)
    roc_auc = metrics.auc(fpr, tpr)     
    ax[0].plot(fpr, tpr, color='darkorange', lw=3, label='area = %0.2f' % roc_auc)
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].hlines(y=recall, xmin=0, xmax=1-cm[0,0]/(cm[0,0]+cm[0,1]), color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[0].vlines(x=1-cm[0,0]/(cm[0,0]+cm[0,1]), ymin=0, ymax=recall, color='red', linestyle='--', alpha=0.7)
    ax[0].set(xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")     
    ax.legend(loc="lower right")
    ax.grid(True)

    **## annotate ROC thresholds**
    thres_in_plot = []
    for i,t in enumerate(thresholds):
         t = np.round(t,1)
         if t not in thres_in_plot:
             ax.annotate(t, xy=(fpr[i],tpr[i]), xytext=(fpr[i],tpr[i]), 
                  textcoords='offset points', ha='left', va='bottom')
             thres_in_plot.append(t)
         else:
             next

    **## plot P-R curve**
    precisions, recalls, thresholds = metrics.**precision_recall_curve**(y_test, predicted_prob)
    roc_auc = metrics.auc(recalls, precisions)
    ax[1].plot(recalls, precisions, color='darkorange', lw=3, label='area = %0.2f' % roc_auc)
    ax[1].plot([0,1], [(cm[1,0]+cm[1,0])/len(y_test), (cm[1,0]+cm[1,0])/len(y_test)], linestyle='--', color='navy', lw=3)
    ax[1].hlines(y=precision, xmin=0, xmax=recall, color='red', linestyle='--', alpha=0.7, label="chosen threshold")
    ax[1].vlines(x=recall, ymin=0, ymax=precision, color='red', linestyle='--', alpha=0.7)
    ax[1].set(xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="lower left")
    ax[1].grid(True)

    **## annotate P-R thresholds
    **thres_in_plot = []
    for i,t in enumerate(thresholds):
        t = np.round(t,1)
        if t not in thres_in_plot:
             ax.annotate(np.round(t,1), xy=(recalls[i],precisions[i]), 
                   xytext=(recalls[i],precisions[i]), 
                   textcoords='offset points', ha='left', va='bottom')
             thres_in_plot.append(t)
        else:
             next
    plt.show()

![](https://cdn-images-1.medium.com/max/2588/1*NXe_UCC-88xREtQ58kovqg.png)

Every point of these curves represents a confusion matrix obtained with a different threshold (the numbers printed on the curves). I could use a threshold of 0.1 and gain a recall of 0.9, meaning that the model would predict 90% of 1s correctly, but the precision would drop to 0.4, meaning that the model would predict a lot of false positives. So it really depends on the type of use case and in particular whether a false positive has an higher cost of a false negative.

When the dataset is balanced and metrics aren’t specified by project stakeholder, I usually choose the threshold that maximize the F1-score. Here’s how:

    **## calculate scores for different thresholds**
    dic_scores = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    XX_train, XX_test, yy_train, yy_test = model_selection.train_test_split(X_train, y_train, test_size=0.2)
    predicted_prob = model.fit(XX_train, yy_train).predict_proba(XX_test)[:,1]

    thresholds = []
    for threshold in np.arange(0.1, 1, step=0.1):
        predicted = (predicted_prob > threshold)
        thresholds.append(threshold)
            dic_scores["accuracy"].append(metrics.accuracy_score(yy_test, predicted))
    dic_scores["precision"].append(metrics.precision_score(yy_test, predicted))
    dic_scores["recall"].append(metrics.recall_score(yy_test, predicted))
    dic_scores["f1"].append(metrics.f1_score(yy_test, predicted))
            
    **## plot
    **dtf_scores = pd.DataFrame(dic_scores).set_index(pd.Index(thresholds))    
    dtf_scores.plot(ax=ax, title="Threshold Selection")
    plt.show()

![](https://cdn-images-1.medium.com/max/2488/1*1EjGMz7axpOjUzLJPtY9dQ.png)

Before moving forward with the last section of this long tutorial, I’d like to say that we can’t say that the model is good or bad yet. The accuracy is 0.85, is it high? Compared to what? You need a **baseline** to compare your model with. Maybe the project you’re working on is about building a new model to replace an old one that can be used as baseline, or you can train different machine learning models on the same train set and compare the performance on a test set.

### Explainability

You analyzed and understood the data, you trained a model and tested it, you’re even satisfied with the performance. You think you’re done? Wrong. High chance that the project stakeholder doesn’t care about your metrics and doesn’t understand your algorithm, so you have to show that your machine learning model is not a black box.

The *Lime *package can help us to build an **explainer**. To give an illustration I will take a random observation from the test set and see what the model predicts:

    print("True:", y_test[4], "--> Pred:", predicted[4], "| Prob:", np.max(predicted_prob[4]))

![](https://cdn-images-1.medium.com/max/2000/1*oosf8SK7ZFcwOuV-rx_4hw.png)

The model thinks that this observation is a 1 with a probability of 0.93 and in fact this passenger did survive. Why? Let’s use the explainer:

    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(y_train), mode="classification")
    explained = explainer.explain_instance(X_test[4], model.predict_proba, num_features=10)
    explained.as_pyplot_figure()

![](https://cdn-images-1.medium.com/max/2050/1*Qk-uB_ckNlatkomYt0U04w.png)

The main factors for this particular prediction are that the passenger is female (Sex_male = 0), young (Age ≤ 22) and traveling in 1st class (Pclass_3 = 0 and Pclass_2 = 0).

The confusion matrix is a great tool to show how the testing went, but I also plot the **classification regions **to give a visual aid of what observations the model predicted correctly and what it missed. In order to plot the data in 2 dimensions some dimensionality reduction is required (the process of reducing the number of features by obtaining a set of principal variables). I will give an example using the [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) algorithm to summarize the data into 2 variables obtained with linear combinations of the features.

    **## PCA**
    pca = decomposition.PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    **## train 2d model**
    model_2d = ensemble.GradientBoostingClassifier()
    model_2d.fit(X_train, y_train)
        
    **## plot classification regions**
    from matplotlib.colors import ListedColormap
    colors = {np.unique(y_test)[0]:"black", np.unique(y_test)[1]:"green"}
    X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=0.01),
    np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))
    fig, ax = plt.subplots()
    Y = model_2d.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    ax.contourf(X1, X2, Y, alpha=0.5, cmap=ListedColormap(list(colors.values())))
    ax.set(xlim=[X1.min(),X1.max()], ylim=[X2.min(),X2.max()], title="Classification regions")
    for i in np.unique(y_test):
        ax.scatter(X_test[y_test==i, 0], X_test[y_test==i, 1], 
                   c=colors[i], label="true "+str(i))  
    plt.legend()
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*gS5PdcX1sk1yQgYQnRSGcg.png)

### Conclusion

This article has been a tutorial to demonstrate **how to approach a classification use case** with data science. I used the Titanic dataset as an example, going through every step from data analysis to the machine learning model.

In the exploratory section, I analyzed the case of a single categorical variable, a single numerical variable and how they interact together. I gave an example of feature engineering extracting a feature from raw data. Regarding preprocessing, I explained how to handle missing values and categorical data. I showed different ways to select the right features, how to use them to build a machine learning classifier and how to assess the performance. In the final section, I gave some suggestions on how to improve the explainability of your machine learning model.

An important note is that I haven’t covered what happens after your model is approved for deployment. Just keep in mind that you need to build a pipeline to automatically process new data that you will get periodically.

Now that you know how to approach a data science use case, you can apply this code and method to any kind of binary classification problem, carry out your own analysis, build your own model and even explain it.
> This article is part of the series **Machine Learning with Python**, see also:
[**Machine Learning with Python: Regression (complete tutorial)**
*Data Analysis & Visualization, Feature Engineering & Selection, Model Design & Testing, Evaluation & Explainability*towardsdatascience.com](https://towardsdatascience.com/machine-learning-with-python-regression-complete-tutorial-47268e546cea)

[**Clustering Geospatial Data**
*Plot Machine Learning & Deep Learning Clustering with interactive Maps*towardsdatascience.com](https://towardsdatascience.com/clustering-geospatial-data-f0584f0b04ec)

