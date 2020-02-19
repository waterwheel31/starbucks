# Starbucks Marketing Campaign Analysis

## Background 

Startbucks is always trying to keep the attention of its customers, and to do so it conducts promotion campaings in regular basis. 

However while some promotion campaigns works well, some campaigns are not successful. This is an intiial analysis of finding the causes. 

Promotion campaings is done by sending offers (such as discounting, buy one get one etc.), through several channels to individual customers. 
The action of cusotmers after sending the offers are as followings

1) Receive offer 
2) View offer
3) Transaction (buy coffee etc.)

The absolute goal is to see how (3) is achieved, but before that (2) viewing is important, since if cusotmers do not see the offers, there wont be any following transactinos. 

In this study, see what fill increase the possibility of the transition from (1) Receiving offer to (2) Viewing offer. 


<br><br>
---- 
## Available Data


For this analysis, we have following data

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record



---- 

<br><br><br>
## Data Exploration 

Before going analysis, lets see how the status of the data is.


<br><br>
#### portfoio.json 

![image](/images/picture1.png)

This is a data about the contents of the 10 different campaings. We can see campaign's `id` is hashed. 

Since some data are not easy to use for machine learning, processed as followings. 

1) Assigned simple numbers to `id` so that it is more readable by human
```
portfolio['new_promo_id'] = portfolio.groupby('id').ngroup() 
```
2) Made `channels` and `offer_type` to one-hot features so that models can understand and also crateded `number of channels` features.  
```
portfolio['email'] = portfolio['channels'].apply(lambda x: True if 'email' in x else False)
portfolio['mobile'] = portfolio['channels'].apply(lambda x: True if 'mobile' in x else False)
portfolio['web'] = portfolio['channels'].apply(lambda x: True if 'web' in x else False)
portfolio['social'] = portfolio['channels'].apply(lambda x: True if 'social' in x else False)
portfolio['n_channel'] = portfolio['channels'].apply(lambda x: len(x))
```
```
portfolio['bogo'] = portfolio['offer_type'].apply(lambda x: True if 'bogo' in x else False)
portfolio['informational'] = portfolio['offer_type'].apply(lambda x: True if 'informational' in x else False)
portfolio['discount'] = portfolio['offer_type'].apply(lambda x: True if 'discount' in x else False)
```



<br><br>

#### profile.json 

this is a data of target users. 
The shape is 17,000 x 5. <br> 
Follwoing is the first 10 data

![image](/images/picture2.png)

User `id` is hashed. 
Also, as we can see from above, there are some missng values. So how many missing values are there? 


<br><br>
```
print(profile.isna().sum())
print(profile.isna().sum()/len(profile))
```
```
missing values
gender              2175
age                    0
id                     0
became_member_on       0
income              2175
dtype: int64
gender              0.127941
age                 0.000000
id                  0.000000
became_member_on    0.000000
income              0.127941
dtype: float64
``` 
Above suggests there are missing values in `gender` and `income` features The numbers of both are the same, and these are linked. Users without `gender` data also lack `income` data. 
In addition to that, from observation above, those people have strange `age` of 118 years old. This sugges a group of strange data is mixed inside. 

Also from the data above, we know those missing data consists about 13% of the whole user data.
 

<br>

```
profile.groupby('id')['id'].count().max()
```
```
> 1
```

There is no duplication in user ID. 


<br><br>



Then, let's see how each feature is distributed. 

##### gender

![image](/images/picture3.png)

a little more male tha female. Also there are some `other` genders.


##### age

![image](/images/picture4.png)

Unimodal distribution, but there is an outlier, at around 118 years old, that is also mentiond earlier. 

##### income

![image](/images/picture5.png)

this shows like a log-normal distribution, as expected


##### became member on 

![image](/images/picture6.png)

Number of recent users are larger, but there are ualso 5 years long users. They may be layal users. 






<br><br>

#### transcript.json 

This is the data of each actionns of users. 

the shape is 306,534 x 12 <br>
following is a random sample of 10 data


![image](/images/picture7.png)

As we can see, there are two types of values missing into the same columns. for `even` is transaction, the `value` data show different values. We need to separate them. Also, the `value` data is dictionary type and needed to be transfomed so that they can be used for analysis. 

```
missing values
person    0
event     0
value     0
time      0
dtype: int64
``` 
There is no missing value in this data


<br>
Then lets see each feature's distribution (showing data counts)

<br>

##### event

![image](/images/picture8.png)

There are more offer received than offer viewed. Meaning not all users do see the offers. 

Also number of transaction is larger than ofer viewed, meaning users do multiple transactions in average after viewing offers. 




##### time 

![image](/images/picture9.png)

We can see some peaks and declines of data 

This can be different by the event type. 

for example, for `offer received` 

![image](/images/picture10.png)

for `offer viewed` 
![image](/images/picture11.png)

This shows that offers are sent periodically and users view with some time lag. 



##### amount (log scale)

![image](/images/picture12.png)



##### events per user 


```
transcript.groupby('person').count().reset_index()['event'].hist()
```
![image](/images/picture13.png)

distributed without bias


<br> 

to use the data for further analysis and especially for machine learnings, following preprocessing are done 

separating `value` column into independent features

```
transcript['value_type'] = transcript['value'].apply(lambda x: list(x.keys())[0])
transcript['value_type'] = transcript['value_type'].apply(lambda x: 'offer_id' if x == 'offer id' else x)
transcript['offer_id'] = transcript['value'].apply(lambda x: list(x.values())[0] if list(x.keys())[0] == 'offer_id' or list(x.keys())[0] == 'offer id' else None )
transcript['offer_id'] = transcript['value'].apply(lambda x: list(x.values())[0] if list(x.keys())[0] == 'offer_id' or list(x.keys())[0] == 'offer id' else None )
transcript['amount'] = transcript['value'].apply(lambda x: list(x.values())[0] if list(x.keys())[0] == 'amount' else None )

```

making categorical value to one-hot feature 
```
transcript['offer_received'] = transcript['event'].apply(lambda x: True if 'offer received' in x else False)
transcript['offer_viewed'] = transcript['event'].apply(lambda x: True if 'offer viewed' in x else False)
transcript['transaction'] = transcript['event'].apply(lambda x: True if 'transaction' in x else False)
transcript['offer_completed'] = transcript['event'].apply(lambda x: True if 'offer completed' in x else False)
```




<br><br>
---- 
## Analysis 


Then let's analyze the causes of what are making users view the offers. 
To analyze that, we see metric named `view rate` as defined below 

```
 view rate = number of offers viewed / numbers of offers received 
```

The problems is to find out what features of the users and the promotions affects the view rate. 

Then let's see the view rates for the 10 promotions 

![image](/images/picture14.png)

Some promotions have close to 100% view rates but some have close to 50% and promotion 0 is less than 40% 

what is making the differnces? That is the problem investigated later.



<br><br>
---

## Approach 

To find out the causes, here we use Logistic Regression classifier. 
This is simple and applicable for this kind of small number data set (around 10K). Also one of its merit compared with other classifiers (ex. Randam Forest, SVM) is it is easy to know what features are contributing. 

For the traing, separate the dataset into training set and test set. Fit the model with the training set and see the results in test set. 

As for the model evaluation, use `accuracy` and also `AUC` considering this is imbalanced case (much more `view` than `non-view)

The benchmark of AUC = 0.50. If the number is larger than 0.50, this means the classifier is meaningful. 


### Process 

first join all the 3 data into one data frame

```
respo = transcript.groupby(['offer_id', 'person'])['offer_received', 'offer_viewed', 'offer_completed'].sum().reset_index()
respo_integ = respo.join(portfolio.set_index('id'), on='offer_id').drop(['offer_id', 'channels', 'offer_type'], axis=1)
respo_integ = respo_integ.join(profile.set_index('id'), on='person').drop('person', axis=1).set_index(['new_promo_id', 'new_user_id'])
respo_integ = respo_integ.fillna(0)
respo_integ = pd.get_dummies(respo_integ)
```

Then take X and y value 
```
X = respo_integ.drop(['offer_viewed', 'offer_received', 'offer_completed'], axis=1)
y = respo_integ['offer_viewed']
```


Scale X data using standard scaler (actually, it is more accurate to scale with trainig X only, but no big impact)
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
```

Dimensionality reduction is not done, since the dimenions are enough small and to avoid losing interpretability. 


Split the data into train and test 
```
X_train, X_test, y_train, y_test = train_test_split( X_scale, y, test_size=0.33, random_state=42)
```

<br>

fit to Logistic Regression classifier

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#clf = RandomForestClassifier(max_depth=5, random_state=0)
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
```


<br><br>
---

## Results

```

pred = clf.predict(X_test)s
acc = accuracy_score(pred, y_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
print('auc:', metrics.auc(fpr, tpr))
```
```
>accuracy: 0.8120750742123911
>auc: 0.6505839773825415
```

AUC is 0.65  <br>
This is not great, but shows this classifier is has classification power. 

Then what features of users or promotions contributed to the `view rate`

![image](/images/picture15.png)

The chart above is the chart of coefficies for each feature. Positive large number means that strongly contributed in positive direction (more view) and negative large number means that are strongly contriubted in negative direction (less view). Numbers close to 0 means no impact. 

From the chart followings are clear, and logically reasonable enough. 

- having many channels is important especially using social media is quite important. 
- Other aspects are not so much important, including the type of offers. 
- users that have been registered for long are likely to view (loyal cusotmers)
- people with high income are more likely to view
- a users of the strange group (age>100, no gender data) are more likely to view. It is not clear what they are from current data, and may worth investigate. 


