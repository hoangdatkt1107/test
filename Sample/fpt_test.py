from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum,
    round,
    regexp_replace,
    count,
    concat,
    lit,
    when,
    rank,
    to_timestamp,
    hour,
    year,
    weekofyear,
    month,
    percentile_approx,
)
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("homework11-10-2023").getOrCreate()

sc = spark.sparkContext

rdd = sc.textFile("../data/FPT_Test/logt*.txt").map(lambda x: eval(x))

log_df = rdd.toDF()
log_df = (
    log_df.select(
        "Mac",
        "SessionMainMenu",
        "AppName",
        "LogId",
        "Event",
        "ItemId",
        "ItemName",
        "RealTimePlaying",
    )
    # Suppose the unit of RealTimePlaying is seconds
    .withColumn("RealTimePlaying", col("RealTimePlaying").cast(FloatType()))
    .withColumnRenamed("Mac", "MAC")
    .withColumnRenamed("LogId", "LogID")
    .withColumnRenamed("ItemId", "ItemID")
)
log_df = log_df.fillna(value=0, subset=["RealTimePlaying"])
# Get Session start time from SessionMainMenu column
log_df = log_df.withColumn(
    "SessionStartTime",
    to_timestamp(
        regexp_replace("SessionMainMenu", "[a-zA-Z0-9]{12}\:", ""),
        "yyyy:MM:dd:HH:mm:ss:SSS",
    ),
).drop("SessionMainMenu")
log_df.printSchema()


user_info_df = (
    spark.read.csv(
        "../data/FPT_Test/user_info.txt",
        sep="\t",
        header=True,
        inferSchema=True,
    )
    .withColumnRenamed("Mac", "MAC")
    .withColumnRenamed("# of days", "nof_days")
    .withColumn(
        "MAC",
        when(
            ~col("MAC").rlike("^(FBOX).*$"), concat(lit("FBOXB"), col("MAC"))
        ).otherwise(col("MAC")),
    )
    .withColumn("MAC", regexp_replace("MAC", "FBOX", ""))
)
# get max nof_days of each user
nof_days_rank_window = Window.partitionBy("MAC").orderBy(col("nof_days").desc())
user_info_df = (
    user_info_df.withColumn("Rank", rank().over(nof_days_rank_window))
    .filter(col("Rank") == 1)
    .drop("Rank")
)
user_info_df.printSchema()


joined_df = user_info_df.join(log_df, user_info_df["MAC"] == log_df["MAC"]).select(
    log_df["*"], user_info_df.nof_days
)
joined_df.printSchema()
joined_df.show(truncate=False)

# Analytics
print(">>> Top 10 users with the most watching time <<<")
top_10_users_with_watching_time_df = (
    joined_df.groupBy("MAC")
    .agg((round(sum("RealTimePlaying"), 2)).alias("total_watching_time"))
    .orderBy("total_watching_time", ascending=False)
    .limit(10)
)
top_10_users_with_watching_time_df.show(truncate=False)

print(">>> Top 10 events <<<")
top_10_events_df = (
    joined_df.groupBy("Event")
    .agg(count("*").alias("event_count"))
    .orderBy("event_count", ascending=False)
    .limit(10)
)
top_10_events_df.show()

print(">>> Patterns in the time of the day and quantity <<<")
time_usage_behaviour_df = (
    joined_df.filter(col("SessionStartTime").isNotNull())
    .withColumn("Hour", hour("SessionStartTime"))
    .select("Hour")
    .groupBy("Hour")
    .agg(count("Hour").alias("Total"))
    .orderBy("Total", ascending=False)
)
time_usage_behaviour_df.show(24)

print(">>> Number of times each user uses the service <<<")
nof_used_times_each_user_df = joined_df.groupBy("MAC").agg(
    count("*").alias("nof_used_times")
)
nof_used_times_each_user_df.orderBy("nof_used_times", ascending=False).show()

print(">>> The average number of used times of all users <<<")
nof_used_times_each_user_df.select(
    percentile_approx("nof_used_times", 0.5).alias("avg_used_times")
).show()


common_df = (
    joined_df.filter(col("SessionStartTime").isNotNull())
    .withColumn("year", year(col("SessionStartTime")))
    .withColumn("week_of_year", weekofyear(col("SessionStartTime")))
    .withColumn("month", month(col("SessionStartTime")))
)
print(">>> The average number of used times of each user in a week <<<")
nof_used_times_in_week_df = (
    common_df.groupBy("MAC", "year", "week_of_year")
    .agg(count("*").alias("count"))
    .groupBy("MAC")
    .agg(percentile_approx("count", 0.5).alias("avg_used_times_in_week"))
)
nof_used_times_in_week_df.orderBy("avg_used_times_in_week", ascending=False).show()

print(">>> The average number of used times of each user in a month <<<")
nof_used_times_in_month_df = (
    common_df.groupBy("MAC", "year", "month")
    .agg(count("*").alias("count"))
    .groupBy("MAC")
    .agg(percentile_approx("count", 0.5).alias("avg_used_times_in_month"))
)
nof_used_times_in_month_df.orderBy("avg_used_times_in_month", ascending=False).show()

print(">>> People with service contracts of less than 1 month <<<")
people_used_less_than_1_month_df = joined_df.filter(col("nof_days") <= 30)
print("> 1. Number of people with service contracts of less than 1 month <")
nof_people_used_less_than_1_month = people_used_less_than_1_month_df.count()
print(nof_people_used_less_than_1_month)
print("> 2. Number of times used by app <")
(
    people_used_less_than_1_month_df.groupBy("AppName")
    .agg(count("*").alias("count_per_group"))
    .withColumn(
        "percent_of_count_total",
        round((col("count_per_group") / nof_people_used_less_than_1_month) * 100, 2),
    )
    .show()
)
print(
    "> 3. Average usage time (average based on RealTimePlaying column) of each user <"
)
(
    people_used_less_than_1_month_df.groupBy("MAC")
    .agg(sum("RealTimePlaying").alias("total_real_time_playing"))
    .agg(
        round(percentile_approx("total_real_time_playing", 0.5), 2).alias(
            "avg_usage_time"
        )
    )
    .show()
)

print(">>> People with service contracts of less than 6 months <<<")
people_used_less_than_6_months_df = joined_df.filter(col("nof_days") <= 180)
print("> 1. Number of people with service contracts of less than 6 months <")
nof_people_used_less_than_6_months = people_used_less_than_6_months_df.count()
print(nof_people_used_less_than_6_months)
print("> 2. Number of times used by app <")
(
    people_used_less_than_6_months_df.groupBy("AppName")
    .agg(count("*").alias("count_per_group"))
    .withColumn(
        "percent_of_count_total",
        round((col("count_per_group") / nof_people_used_less_than_6_months) * 100, 2),
    )
    .show()
)
print(
    "> 3. Average usage time (average based on RealTimePlaying column) of each user <"
)
(
    people_used_less_than_6_months_df.groupBy("MAC")
    .agg(sum("RealTimePlaying").alias("total_real_time_playing"))
    .agg(
        round(percentile_approx("total_real_time_playing", 0.5), 2).alias(
            "avg_usage_time"
        )
    )
    .show()
)

print(">>> People with service contracts of more than 1 year <<<")
people_used_more_than_1_year_df = joined_df.filter(col("nof_days") >= 365)
print("> 1. Number of people with service contracts of more than 1 year <")
nof_people_used_more_than_1_year = people_used_more_than_1_year_df.count()
print(nof_people_used_more_than_1_year)
print("> 2. Number of times used by app <")
(
    people_used_more_than_1_year_df.groupBy("AppName")
    .agg(count("*").alias("count_per_group"))
    .withColumn(
        "percent_of_count_total",
        round((col("count_per_group") / nof_people_used_more_than_1_year) * 100, 2),
    )
    .show()
)
print(
    "> 3. Average usage time (average based on RealTimePlaying column) of each user <"
)
(
    people_used_more_than_1_year_df.groupBy("MAC")
    .agg(sum("RealTimePlaying").alias("total_real_time_playing"))
    .agg(
        round(percentile_approx("total_real_time_playing", 0.5), 2).alias(
            "avg_usage_time"
        )
    )
    .show()
)
