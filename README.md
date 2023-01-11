# Predict-Future-Sales

Sales forecasting is a frequent application of Machine Learning. Businesses can use this forecasting to identify benchmarks, determine incremental impacts of new initiatives, plan resources in response to expected demand, and project future budgets. This report provides detailed Machine Learning based solutions to the <a href="https://www.kaggle.com/c/competitive-data-science-predict-future-sales">Kaggle competition - Predict Future Sales </a>. I have implemented different Machine Learning models to produce forecasted output for the given dataset and concluded the best results with Kaggle ranking. 

<h3>Problem Statement and Methods applied</h3>
<p>The task is to forecast the total amount of products sold in every shop for the test set. We have applied following models:
<ol>
<li>Prophet</li>
<li>LSTM</li>
<li>ARIMA</li>
<li>LightGBM</li>
<li>XGBoost</li>
</ol>

Check out detailed report of each model implemented <a href="https://docs.google.com/document/d/1JJp8aekJ470IJNwLytv9RFhP0BZMUtIeDTuWqgZEP3c/edit">here</a>.

<h3>Data Description</h3>
<p>The dataset used is time-series daily historical sales data. It consists of <b>2,170 items</b> sold by <b>60 shops</b> between <b>January 2013 to October 2015</b>.</p> 
<p>The data set consist of 6 csv extension files which are given below with their descriptions: </p>
<ol>
<li><b>sales_train.csv:</b> This is the training set which consist of historical data from January 2013 to October 2015.</li>
<li><b>test.csv:</b> This is the test set. We are expected to forecast the sales for these given shops and products for November 2015.</li>
<li><b>sample_submission.csv:</b> This file exhibits the correct format expected.</li>
<li><b>items.csv:</b> supplemental information about the items categories.</li>
<li><b>shops.csv:</b> supplemental information about the shops.</li>
</ol>
	
The Data fields present in these with their descriptions are given below:
<ol>
<li><b>ID -</b> an Id that represents a (Shop, Item) tuple within the test set
<li><b>shop_id -</b> unique identifier of a shop
<li><b>item_id -</b> unique identifier of a product
<li><b>item_category_id -</b> unique identifier of item category
<li><b>item_cnt_day -</b> number of products sold. You are predicting a monthly amount of this measure
<li><b>item_price -</b> current price of an item
<li><b>date -</b> date in format dd/mm/yyyy
<li><b>date_block_num -</b> a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1 , . . . , October 2015 is 33
<li><b>item_name -</b> name of item
<li><b>shop_name - </b>name of shop
<li><b>item_category_name -</b> name of item category
</ol>
<p>The dataset can be download from <a href="https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data">here</a>.

<h3>Objective</h3>
<p>The competition requires us to predict the future total sales to happen in the next month (Nov 2015) for every item and store in the test file. 
<p>The submissions are evaluated by root mean squared error (RMSE) with true target values clipped into [0,20] range.

<h3>Litrature Review</h3>
<h4>Why predict future sales?</h4>
<p>Sales forecasting is a technique that uses historical sales data as inputs to make informed predictions about the direction of future trends.
<ul>
<li><b>Manage supply chain efficiently:</b> Knowing future consumer trends allow business’ sales operations align their supply chain activities efficiently like material purchases, inventory stocking, warehouse capacity plans, hiring and handle the market demand most efficiently with improvised decision making. 
<li><b>Make higher Revenue:</b> It enables companies to focus their sales team on high-profit sales opportunities resulting in higher revenue.
<li><b>Incorporate right changes:</b> It enables companies to incorporate the right changes like pricing, marketing, product changes, locations, hiring etc for improved business outcomes.
</ul>
<h4>Different Sales Forecasting Techniques:</h4>
<ol>
<li>Qualitative Methods:</li>
<ul>
<li><b>Market Research:</b> A systemic process of actively surveying or interviewing potential customers to determine the interest of service or product.</li>
<li><b>Delphi Method: </b>A panel of experts is interviewed by a sequence of questionnaires enabling forecaster to have all information for forecasting.</li>
<li><b>Visionary Forecast:</b> It is a non-scientific method where a 'visionary' or 'futurologist' attempt to forecast through subjective opinion, guesswork and imagination.</li>
</ul>
<li>Time Series Analysis and Projection:</li>
<ul>
<li><b>Moving Average:</b> It is technical indicator that investors and traders use to determine trend direction, and seasonal irregularities. It is calculated by adding up data points during specific period and dividing by number of time periods.</li>
<li><b>Exponential smoothing:</b> This is similar to moving average except that more recent data points are given more weight. Applied mainly for production and inventory control.</li>
<li><b>Trend Projection:</b> This technique fits a trend line to a mathematical equation and then projects it into the future by means of this equation. It is typically used to forecast new-products and long-term sales.</li>
</ul>
<li>Casual methods:</li>
<ul>
<li><b>Regression model:</b> This functionally relates sales to other economic or internal variables to estimate an equation using least-square error technique. It is good for short-term predictions.</li>
<li><b>Life-cycle Analysis:</b> The product acceptance by various groups is analysed to forecast product growth rates.</li>
</ul>
</ol>
<h3>Data Exploration and Data Pre-processing:</3>
<h4>Data Exploration:</h4>
<p>The very first step in data analysis is to explore and visualise the unstructured data to uncover patterns, characteristics, and points of interest. It creates a broader picture of important trends and points that require further study. It also gives us an idea of the amount of cleaning required in the data. Given below are some glimpses of data set.<p>
<p>Since the original dataset was in Russian language, in order to better understand the data and to see if there is any scope for feature extraction, it is translated to English using translation tools</p>

<img src="https://user-images.githubusercontent.com/41836325/176777406-33dd8f3b-cbe0-4cae-92b0-93484fa1500c.png" width="55%">
<img src="https://user-images.githubusercontent.com/41836325/176778112-76ad834b-8224-4f7e-a616-37e0692ee51e.png" width="50%">
<img src="https://user-images.githubusercontent.com/41836325/176778647-a6efc1e5-7752-4668-b83e-deb69d57cd1e.png" width="50%">
<img src="https://user-images.githubusercontent.com/41836325/176778376-d1bc498c-0e06-4559-819b-7ed4cce81469.png" width="40%">

<p>From exploration we can conclude that we only need to forecast sales for 5,100 items for 42 shops. Hence, we may not include all shops and items to reduce computing resources required to train models.</p>

<h3>Visualizing Data</h3>
<p>For data visualization, we have mainly used Plotly, Seaborn and Matplotlib python visualization libraries.

<b>1. Distribution of items in each category:</b> We can see the distribution of items among 83 categories. Item category 40 has highest number of items.

<img src="https://user-images.githubusercontent.com/41836325/176785731-a16993e2-66df-4b57-bf94-4555943525b1.png" width="85%">

<b>2. Total sales made by each shop over the span of 34 months. </b>

<img src="https://user-images.githubusercontent.com/41836325/176786045-186c08f6-8ccb-4ec8-91d6-92d5a6c02ce9.png" width="85%">

3.Below plot shows the <b>total number of items sold by each shop on a day</b> over the span of 34 months. We can see that shop no. 9 opens sporadically and makes huge sales when opened. We also see peaks in the year ends sales.

<img src="https://user-images.githubusercontent.com/41836325/176786200-40dc2d1f-a3aa-4268-b198-586003a3784f.png" width="85%">

<b>4. Plot of item prices of each item:</b> We can most items are pretty much in same price range except for one item (6606) which is way high. This is clearly an outlier.

<img src="https://user-images.githubusercontent.com/41836325/176786427-ffc9c68e-c7a5-4bb9-8db7-12bc10bea785.png" width="85%">

<b>5. Below plot shows the total sales and number of items sold in a month:</b> We can see there is seasonality in the sales trend. The sales seem to peak in the year end, and then follows a decreasing trend.

<img src="https://user-images.githubusercontent.com/41836325/176786516-6f03ecb3-bb68-4a80-9031-83e1e5d6d8bd.png" width="85%">

<h3>Data cleaning:</h3>
<p>Data in its true form is raw and not usable. It needs to be cleaned and produced in a form that is more readable and usable. The practice of modifying or altering data in order to make it more understandable and structured is known as data manipulation. It enhances the quality of the data for future modelling purposes. Following are some of the steps performed to clean the data
<p>From above plots, we can see there are clearly some outliers that needs to be treated.
<p>An outlier is an observation or value that lies an abnormal distance from other values in a given sample. These are stranglers that can be extremely high values or extremely low values. This can be variability in the measurement or it can sometimes indicate an error during experiment. Usually, outliers can lead to misleading interpretations and hence are advised to be removed before training a model. 

<p>Detecting outliers using box plot.</p>
<b>1. Box plot of item_price feature:</b> We see there is one particular item having price above 300k, far away from rest of the sample.
<img src="https://user-images.githubusercontent.com/41836325/176786960-935d235f-8b84-4fc5-822d-239f8b2e7059.png" width="60%">

<b>2. Box plot of item_cnt_day feature:</b> We see there is one sample with item count more than 2000.
<img src="https://user-images.githubusercontent.com/41836325/176787089-da6d9863-918a-421a-a800-831f20f61379.png" width="60%">

<p>We simply remove these outlier samples as they can skew the training considerably</p>
<img src="https://user-images.githubusercontent.com/41836325/176787154-3fcb4819-cbfa-4823-9c84-60b12119d9ec.png" width="30%">

<p><b>3. Handling negative values:</b>  We see train data has samples with negative item_price and negative item count.</p>
<img src="https://user-images.githubusercontent.com/41836325/176787231-0b739e7b-f37a-4177-baec-898862ca817f.png" width="20%">

<p>Since item price and count is not fixed and vary with months, we have handled them by making the negative values to null and imputing them using Scikit-learn’s <b>KNNImputer</b> that uses K-Nearest Neighbour algorithm to assign null values with values of it’s closes neighbouring sample.</p>
<img src="https://user-images.githubusercontent.com/41836325/176787868-3234b4e4-da55-454f-835f-150e49678262.png" width="45%">

<p><b>4.Handling null values:</b> This dataset has no null values.</p>
<img src="https://user-images.githubusercontent.com/41836325/176788118-eaa01c3a-8bac-4bdb-92c7-515f9b18ad5f.png" width="20%">

<h3>Problem Statement and Methods applied</h3>
<p>Our task is to forecast the total amount of products sold in every shop for the test set. We had applied following models:
<ol>
<li>Prophet</li>
<li>LSTM</li>
<li>ARIMA</li>
<li>LightGBM</li>
<li>XGBoost</li>
</ol>

Check out detailed report of each model implemented <a href="https://docs.google.com/document/d/1JJp8aekJ470IJNwLytv9RFhP0BZMUtIeDTuWqgZEP3c/edit">here</a>.
