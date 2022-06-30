# Predict-Future-Sales

Sales forecasting is a frequent application of Machine Learning. Businesses can use this forecasting to identify benchmarks, determine incremental impacts of new initiatives, plan resources in response to expected demand, and project future budgets. This report provides detailed Machine Learning based solutions to the <a href="https://www.kaggle.com/c/competitive-data-science-predict-future-sales">Kaggle competition - Predict Future Sales </a>. We have implemented different Machine Learning models to produce forecasted output for the given dataset and concluded the best results with Kaggle ranking. 

<h3>Data Description</h3>
<p>The dataset provided to us is a time-series daily historical sales data. It consists of <b>2,170 items</b> sold by <b>60 shops</b> between <b>January 2013 to October 2015</b>.</p> 
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
![image](https://user-images.githubusercontent.com/41836325/176776976-7f724182-1d99-465f-a545-e5b48ea0f149.png)

