from pathlib import Path
import seaborn as sns
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
DATA_DIR = Path("/Users/allisonlu/242proj2/Fitabase Data 4.12.16-5.12.16")

# 1 - distributions and outliers
def make_date_axis(ax):
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()

# heart rate
heartrate = pd.read_csv(DATA_DIR/"heartrate_seconds_merged.csv")
heartrate["Time"] = pd.to_datetime(heartrate["Time"], errors="coerce")
heartrate["Date"] = heartrate["Time"].dt.floor("D")
hr_daily = (heartrate.groupby(["Id", "Date"], as_index=False)["Value"]
             .mean()
             .rename(columns={"Value": "AvgHeartRate"}))

top5_heartrate = hr_daily["Id"].value_counts().head(5).index.tolist()
heartrate_ids = hr_daily[hr_daily["Id"].isin(top5_heartrate)]

fig, ax = plt.subplots(figsize=(10, 5))
for uid, sub in heartrate_ids.groupby("Id"):
    sub = sub.sort_values("Date")
    ax.plot(sub["Date"], sub["AvgHeartRate"], label=str(uid))

ax.set_title("Daily Average Heart Rate for 5 Users")
ax.set_xlabel("Date")
ax.set_ylabel("Average Heart Rate (bpm)")
ax.legend(title="User Id", ncols=2, fontsize=8)
make_date_axis(ax)
plt.show()

# sleep
sleep_duration = pd.read_csv(DATA_DIR/"sleepDay_merged.csv")
sleep_duration["Date"] = pd.to_datetime(sleep_duration["SleepDay"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

top5_sleep_ids = sleep_duration["Id"].value_counts().head(5).index.tolist()
sleep5 = sleep_duration[sleep_duration["Id"].isin(top5_sleep_ids)].copy()
sleep5 = sleep5.sort_values(["Id", "Date"])

fig, ax = plt.subplots(figsize=(10, 5))
for uid, sub in sleep5.groupby("Id"):
    ax.plot(sub["Date"], sub["TotalMinutesAsleep"], label=str(uid))

ax.set_title("Daily Sleep Duration for 5 Users")
ax.set_xlabel("Date")
ax.set_ylabel("Total Minutes Asleep")
ax.legend(title="User Id", ncols=2, fontsize=8)
make_date_axis(ax)
plt.show()

# daily activity
daily_activity = pd.read_csv(DATA_DIR/"dailyActivity_merged.csv")
daily_activity["Date"] = pd.to_datetime(daily_activity["ActivityDate"],errors="coerce")

top5_steps_ids = daily_activity["Id"].value_counts().head(5).index.tolist()
steps5 = daily_activity[daily_activity["Id"].isin(top5_steps_ids)].copy()
steps5 = steps5.sort_values(["Id", "Date"])

fig, ax = plt.subplots(figsize=(10, 5))
for uid, sub in steps5.groupby("Id"):
    ax.plot(sub["Date"], sub["TotalSteps"], label=str(uid))

ax.set_title("Daily Steps for 5 Users")
ax.set_xlabel("Date")
ax.set_ylabel("Total Steps")
ax.legend(title="User Id", ncols=2, fontsize=8)
make_date_axis(ax)
plt.show()

# weight
weight = pd.read_csv(DATA_DIR/"weightLogInfo_merged.csv")
weight["Date"] = pd.to_datetime(weight["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

id_most_weight = weight["Id"].value_counts().idxmax()
wuser = weight[weight["Id"] == id_most_weight].copy()
wuser_daily = (
    wuser.groupby("Date", as_index=False)["WeightKg"]
         .mean()
         .sort_values("Date")
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wuser_daily["Date"], wuser_daily["WeightKg"])
ax.set_title(f"Weight Change Over Time (User {id_most_weight})")
ax.set_xlabel("Date")
ax.set_ylabel("Weight (kg)")
make_date_axis(ax)
plt.show()


# 2 data processing
# part 1 - hourly and minutely dataframe
hourly_steps = pd.read_csv(DATA_DIR/"hourlySteps_merged.csv")
hourly_int   = pd.read_csv(DATA_DIR/"hourlyIntensities_merged.csv")
hourly_cal   = pd.read_csv(DATA_DIR/"hourlyCalories_merged.csv")

hourly_df = (hourly_steps
          .merge(hourly_int, on=["Id", "ActivityHour"])
          .merge(hourly_cal, on=["Id", "ActivityHour"]))

min_cal  = pd.read_csv(DATA_DIR/"minuteCaloriesNarrow_merged.csv")
min_int  = pd.read_csv(DATA_DIR/"minuteIntensitiesNarrow_merged.csv")
min_mets = pd.read_csv(DATA_DIR/"minuteMETsNarrow_merged.csv")

minutely_df = (min_cal
            .merge(min_int, on=["Id", "ActivityMinute"])
            .merge(min_mets, on=["Id", "ActivityMinute"]))

print(hourly_df.head())
print(minutely_df.head())


# part 2 - convert time to datetime
daily_activity = pd.read_csv(DATA_DIR/"dailyActivity_merged.csv")

daily_activity["Date"] = pd.to_datetime(daily_activity["ActivityDate"], format="%m/%d/%Y", errors="coerce")
sleep_duration["Date"] = pd.to_datetime(sleep_duration["SleepDay"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
hourly_df["Date"] = pd.to_datetime(hourly_df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
minutely_df["Date"] = pd.to_datetime(minutely_df["ActivityMinute"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

print(daily_activity.head())
print(sleep_duration.head())
print(hourly_df.head())
print(minutely_df.head())

# part 3 - process heart rate data
heartrate["Time"] = pd.to_datetime(heartrate["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
heartrate["Minute"] = heartrate["Time"].dt.floor("min")
minutely_df["Minute"] = minutely_df["Date"].dt.floor("min")

heartrate_min = (heartrate.groupby(["Id", "Minute"], as_index=False)["Value"].mean()
              .rename(columns={"Value": "Avg Heart Rate"}))

minutely_heartrate = minutely_df.merge(
    heartrate_min,
    on=["Id", "Minute"],
    how="inner"
)

# print(minutely_heartrate.head())


# 3 correlation and processing
# part 1 - scatterplots between step vs cal, intensity vs cal
hourly_cals = pd.read_csv(DATA_DIR/"hourlyCalories_merged.csv")
hourly_int = pd.read_csv(DATA_DIR/"hourlyIntensities_merged.csv")
hourly_steps = pd.read_csv(DATA_DIR/"hourlySteps_merged.csv")

sns.scatterplot(data=hourly_df, x="StepTotal", y="Calories")
plt.title("Steps vs Calories")
plt.xlabel("Steps")
plt.ylabel("Calories Burned")
plt.show()

sns.scatterplot(data=hourly_df, x="TotalIntensity", y="Calories")
plt.title("Intensity vs Calories")
plt.xlabel("Total Intensity")
plt.ylabel("Calories Burned")
plt.show()

print(hourly_df[["StepTotal", "TotalIntensity", "Calories"]].corr())


# part 2 - total minutes asleep vs total minutes in bed
sns.scatterplot(data=sleep_duration, x="TotalTimeInBed", y="TotalMinutesAsleep")
plt.title("Sleep Duration vs Time in Bed")
plt.xlabel("Total Time in Bed (minutes)")
plt.ylabel("Total Minutes Asleep")
plt.show()
print(sleep_duration[["TotalTimeInBed", "TotalMinutesAsleep"]].corr())

# part 3 - distribution of intensity over a day
hourly_df["Hour"] = pd.to_datetime(hourly_df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p").dt.hour
avg_intensity = hourly_df.groupby(["Id", "Hour"])["TotalIntensity"].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=avg_intensity, x="Hour", y="TotalIntensity", hue="Id", errorbar=None)
plt.title("Average Intensity by Hour for Each User")
plt.xlabel("Hour of Day")
plt.ylabel("Average Intensity")
plt.legend(title="User ID")
plt.tight_layout()
plt.show()

# part 4 - total minutes asleep vs total sedentary minutes
daily_sleep = daily_activity.merge(
    sleep_duration[["Id", "Date", "TotalMinutesAsleep"]],
    on=["Id", "Date"],
    how="inner"
)

sns.scatterplot(data=daily_sleep, x="SedentaryMinutes", y="TotalMinutesAsleep")
plt.title("Sedentary Time vs Sleep Duration")
plt.xlabel("Sedentary Minutes per Day")
plt.ylabel("Total Minutes Asleep")
plt.show()

print(daily_sleep[["SedentaryMinutes", "TotalMinutesAsleep"]].corr())


# 4 T-tests
# most significant difference - intensity and calories
# least significat - sleep and calories

daily_activity["IntensityScore"] = (
    daily_activity["VeryActiveMinutes"] * 3 +
    daily_activity["FairlyActiveMinutes"] * 2 +
    daily_activity["LightlyActiveMinutes"] * 1
)

daily_sleep = daily_activity.merge(
    sleep_duration[["Id", "Date", "TotalMinutesAsleep"]],
    on=["Id", "Date"],
    how="inner"
).dropna(subset=["Calories", "IntensityScore", "TotalMinutesAsleep"])

intensity_med = daily_sleep["IntensityScore"].median()
sleep_med = daily_sleep["TotalMinutesAsleep"].median()

high_intensity = daily_sleep[daily_sleep["IntensityScore"] > intensity_med]["Calories"]
low_intensity  = daily_sleep[daily_sleep["IntensityScore"] <= intensity_med]["Calories"]

high_sleep = daily_sleep[daily_sleep["TotalMinutesAsleep"] > sleep_med]["Calories"]
low_sleep  = daily_sleep[daily_sleep["TotalMinutesAsleep"] <= sleep_med]["Calories"]

t_intensity, p_intensity = stats.ttest_ind(high_intensity, low_intensity, equal_var=False)
t_sleep, p_sleep = stats.ttest_ind(high_sleep, low_sleep, equal_var=False)

print("Intensity Test: t =", t_intensity, " p =", p_intensity)
print("Sleep Duration Test: t =", t_sleep, " p =", p_sleep)






