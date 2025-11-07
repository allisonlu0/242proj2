from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates

DATA_DIR = Path("/Users/allisonlu/242proj2/Fitabase Data 4.12.16-5.12.16")

# loading data
heartrate = pd.read_csv(DATA_DIR/"heartrate_seconds_merged.csv")
sleep_duration = pd.read_csv(DATA_DIR/"sleepDay_merged.csv")
daily_steps = pd.read_csv(DATA_DIR/"dailySteps_merged.csv")
weight = pd.read_csv(DATA_DIR/"weightLogInfo_merged.csv")

# pip install pandas matplotlib numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------
# helpers for date axes & outliers
# -----------------------
def make_date_axis(ax):
    """Readable date axis: auto locator + concise formatter + rotated labels."""
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.figure.tight_layout()

def zscore_outliers(s, thresh=3.0):
    """Return boolean mask of z>thresh, ignoring NaNs."""
    s = s.astype(float)
    z = (s - s.mean()) / s.std(ddof=1)
    return z.abs() > thresh

# -----------------------
# read data
# -----------------------

# heart rate (seconds) -> parse & aggregate to daily mean
heartrate["Time"] = pd.to_datetime(heartrate["Time"], errors="coerce")
heartrate["Date"] = heartrate["Time"].dt.floor("D")
hr_daily = (heartrate
            .groupby(["Id", "Date"], as_index=False)["Value"]
            .mean()
            .rename(columns={"Value": "AvgHeartRate"}))

# sleep (daily)
sleep_duration["Date"] = pd.to_datetime(sleep_duration["SleepDay"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# daily activity (steps, etc.)
daily_activity = pd.read_csv(DATA_DIR/"dailyActivity_merged.csv")
daily_activity["Date"] = pd.to_datetime(daily_activity["ActivityDate"],errors="coerce")

# weight


# common columns are "Date" or "DateTime"; normalize to "Date"
if "Date" in weight.columns:
    weight["Date"] = pd.to_datetime(weight["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
elif "DateTime" in weight.columns:
    weight["Date"] = pd.to_datetime(weight["DateTime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
else:
    # fallback for some datasets
    first_dt_col = next((c for c in weight.columns if "date" in c.lower()), None)
    weight["Date"] = pd.to_datetime(weight[first_dt_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# -----------------------
# (a) Heart rate for 5 different users (x-axis = date)
# pick 5 users with most HR days
# -----------------------
top5_heartrate = (hr_daily["Id"].value_counts().head(5).index.tolist())
heartrate_ids = hr_daily[hr_daily["Id"].isin(top5_heartrate)].copy()

fig, ax = plt.subplots(figsize=(10, 5))
for uid, sub in heartrate_ids.groupby("Id"):
    sub = sub.sort_values("Date")
    ax.plot(sub["Date"], sub["AvgHeartRate"], label=str(uid))
    # flag outliers in-line (z>3)
    mask = zscore_outliers(sub["AvgHeartRate"])
    if mask.any():
        ax.scatter(sub.loc[mask, "Date"], sub.loc[mask, "AvgHeartRate"])

ax.set_title("Daily Average Heart Rate for 5 Users")
ax.set_xlabel("Date")
ax.set_ylabel("Average Heart Rate (bpm)")
ax.legend(title="User Id", ncols=2, fontsize=8)
make_date_axis(ax)

# -----------------------
# (b) Daily sleep duration for 5 different users
# -----------------------
top5_sleep_ids = (sleep_duration["Id"].value_counts().head(5).index.tolist())
sleep5 = sleep_duration[sleep_duration["Id"].isin(top5_sleep_ids)].copy()
sleep5 = sleep5.sort_values(["Id", "Date"])

fig, ax = plt.subplots(figsize=(10, 5))
for uid, sub in sleep5.groupby("Id"):
    ax.plot(sub["Date"], sub["TotalMinutesAsleep"], label=str(uid))
    mask = zscore_outliers(sub["TotalMinutesAsleep"])
    if mask.any():
        ax.scatter(sub.loc[mask, "Date"], sub.loc[mask, "TotalMinutesAsleep"])

ax.set_title("Daily Sleep Duration for 5 Users")
ax.set_xlabel("Date")
ax.set_ylabel("Total Minutes Asleep")
ax.legend(title="User Id", ncols=2, fontsize=8)
make_date_axis(ax)

# -----------------------
# (c) Daily steps for 5 different users
# -----------------------
top5_steps_ids = (daily_activity["Id"].value_counts().head(5).index.tolist())
ids_random_steps = np.random.choice(daily_activity["Id"].unique(), 5, replace=False)
steps5 = daily_activity[daily_activity["Id"].isin(ids_random_steps)].copy()
steps5 = steps5.sort_values(["Id", "Date"])

fig, ax = plt.subplots(figsize=(10, 5))
for uid, sub in steps5.groupby("Id"):
    ax.plot(sub["Date"], sub["TotalSteps"], label=str(uid))
    mask = zscore_outliers(sub["TotalSteps"])
    if mask.any():
        ax.scatter(sub.loc[mask, "Date"], sub.loc[mask, "TotalSteps"])

ax.set_title("Daily Steps for 5 Users")
ax.set_xlabel("Date")
ax.set_ylabel("Total Steps")
ax.legend(title="User Id", ncols=2, fontsize=8)
make_date_axis(ax)

# -----------------------
# (d) Weight change of the user with the highest number of weight records
# -----------------------
id_most_weight = (weight["Id"].value_counts().idxmax())
wuser = weight[weight["Id"] == id_most_weight].copy()
# If multiple logs per day, average to daily to smooth
wuser_daily = (wuser.groupby("Date", as_index=False)["WeightKg"].mean()
                     .sort_values("Date"))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wuser_daily["Date"], wuser_daily["WeightKg"])
mask = zscore_outliers(wuser_daily["WeightKg"])
if mask.any():
    ax.scatter(wuser_daily.loc[mask, "Date"], wuser_daily.loc[mask, "WeightKg"])

ax.set_title(f"Weight Change Over Time (User {id_most_weight})")
ax.set_xlabel("Date")
ax.set_ylabel("Weight (kg)")
make_date_axis(ax)

plt.show()

# -----------------------
# Outlier notes for your write-up (optional console prints)
# -----------------------
def print_outlier_summary(df, date_col, value_col, label):
    out = []
    for uid, sub in df.groupby("Id"):
        m = zscore_outliers(sub[value_col])
        if m.any():
            rows = sub.loc[m, [date_col, value_col]]
            out.append((uid, rows))
    if out:
        print(f"\nPossible outliers in {label}:")
        for uid, rows in out:
            for _, r in rows.iterrows():
                print(f"  Id {uid}: {r[date_col].date()} -> {value_col}={r[value_col]}")
    else:
        print(f"\nNo strong outliers detected in {label} (z>3).")

print_outlier_summary(heartrate_ids.rename(columns={"AvgHeartRate":"val", "Date":"d"}), "d", "val", "heart rate")
print_outlier_summary(sleep5.rename(columns={"TotalMinutesAsleep":"val", "Date":"d"}), "d", "val", "sleep")
print_outlier_summary(steps5.rename(columns={"TotalSteps":"val", "Date":"d"}), "d", "val", "steps")

if zscore_outliers(wuser_daily["WeightKg"]).any():
    print("\nWeight plot: surprising outliers present (z>3). Inspect above scatter markers.")
else:
    print("\nWeight plot: no strong outliers (z>3).")


pd.set_option('display.max_columns', None)


''''''
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

'''print(hourly_df.head())
print(minutely_df.head())'''

# part 2 - convert time to datetime
daily_activity = pd.read_csv(DATA_DIR/"dailyActivity_merged.csv")

daily_activity["Date"] = pd.to_datetime(daily_activity["ActivityDate"], format="%m/%d/%Y", errors="coerce")
sleep_duration["Date"] = pd.to_datetime(sleep_duration["SleepDay"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
hourly_df["Date"] = pd.to_datetime(hourly_df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
minutely_df["Date"] = pd.to_datetime(minutely_df["ActivityMinute"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

'''print(daily_activity.head())
print(sleep_duration.head())
print(hourly_df.head())
print(minutely_df.head())'''

# part 3 - process heart rate data
heartrate["Time"] = pd.to_datetime(heartrate["Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

# print(heartrate.head())

# Floor both to minute to align timestamps
heartrate["Minute"] = heartrate["Time"].dt.floor("min")
minutely_df["Minute"] = minutely_df["Date"].dt.floor("min")

heartrate_min = (heartrate.groupby(["Id", "Minute"], as_index=False)["Value"].mean()
              .rename(columns={"Value": "Avg Heart Rate"}))

minutely_heartrate = minutely_df.merge(
    heartrate_min,
    on=["Id", "Minute"],
    how="inner"
)

'''OR
minutely_df["_MinuteKey"] = minutely_df["Date"].dt.floor("min")

minutely_with_hr = minutely_df.merge(
    hr_per_min,
    left_on=["Id", "_MinuteKey"],
    right_on=["Id", "Minute"],
    how="left"
).drop(columns=["_MinuteKey", "Minute"])
'''

# print(minutely_heartrate.head())


# 3 correlation and processing
# part 1 - scatterplots between step vs cal, intensity vs cal
hourly_cals = pd.read_csv(DATA_DIR/"hourlyCalories_merged.csv")
hourly_int = pd.read_csv(DATA_DIR/"hourlyIntensities_merged.csv")
hourly_steps = pd.read_csv(DATA_DIR/"hourlySteps_merged.csv")

sns.scatterplot(data=hourly_df, x="StepTotal", y="Calories")
plt.title("Steps vs Calories (Hourly)")
plt.xlabel("Steps")
plt.ylabel("Calories Burned")
plt.show()

sns.scatterplot(data=hourly_df, x="TotalIntensity", y="Calories")
plt.title("Intensity vs Calories (Hourly)")
plt.xlabel("Total Intensity")
plt.ylabel("Calories Burned")
plt.show()

print(hourly_df[["StepTotal", "TotalIntensity", "Calories"]].corr())
# intensity has stronger correlation with calories but not by that much
# Intensity vs Calories is often more strongly correlated, because calories are
# a direct function of energy expenditure intensity rather than just step count.

# part 2 - total minutes asleep vs total minutes in bed
sns.scatterplot(data=sleep_duration, x="TotalTimeInBed", y="TotalMinutesAsleep")
plt.title("Sleep Duration vs Time in Bed")
plt.xlabel("Total Time in Bed (minutes)")
plt.ylabel("Total Minutes Asleep")
plt.show()

# part 3 - distribution of intensity over a day
hourly_df["Hour"] = pd.to_datetime(hourly_df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p").dt.hour
avg_intensity = hourly_df.groupby(["Id", "Hour"])["TotalIntensity"].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=avg_intensity, x="Hour", y="TotalIntensity", hue="Id", errorbar=None)
plt.title("Average Intensity by Hour for Each User")
plt.xlabel("Hour of Day")
plt.ylabel("Average Intensity")
plt.legend(title="User ID", bbox_to_anchor=(1.05, 1), loc='upper left')
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
# there is a negative relationship but why

# 4 T-tests
# most significant difference - intensity and calories
'''
A) Calories vs IntensityScore
H₀: There is no difference in mean calories burned between high-intensity and low-intensity days.
H₁: There is a difference in mean calories burned between high-intensity and low-intensity days.

B) Calories vs TotalMinutesAsleep
H₀: There is no difference in mean calories burned between days with high and low sleep duration.
H₁: There is a difference in mean calories burned between days with high and low sleep duration.
'''

import pandas as pd
from scipy import stats

# --- Create IntensityScore in daily_activity ---
daily_activity["IntensityScore"] = (
    daily_activity["VeryActiveMinutes"] * 3 +
    daily_activity["FairlyActiveMinutes"] * 2 +
    daily_activity["LightlyActiveMinutes"] * 1
)

# --- Merge to include sleep with calories + intensity ---
daily_sleep = daily_activity.merge(
    sleep_duration[["Id", "Date", "TotalMinutesAsleep"]],
    on=["Id", "Date"],
    how="inner"
).dropna(subset=["Calories", "IntensityScore", "TotalMinutesAsleep"])

# --- Split groups by median (High vs Low) ---
intensity_med = daily_sleep["IntensityScore"].median()
sleep_med = daily_sleep["TotalMinutesAsleep"].median()

# Intensity groups for Calories
high_intensity = daily_sleep[daily_sleep["IntensityScore"] > intensity_med]["Calories"]
low_intensity  = daily_sleep[daily_sleep["IntensityScore"] <= intensity_med]["Calories"]

# Sleep groups for Calories
high_sleep = daily_sleep[daily_sleep["TotalMinutesAsleep"] > sleep_med]["Calories"]
low_sleep  = daily_sleep[daily_sleep["TotalMinutesAsleep"] <= sleep_med]["Calories"]

# --- Two-sample Independent t-tests ---
t_intensity, p_intensity = stats.ttest_ind(high_intensity, low_intensity, equal_var=False)
t_sleep, p_sleep = stats.ttest_ind(high_sleep, low_sleep, equal_var=False)

print("Intensity Test: t =", t_intensity, " p =", p_intensity)
print("Sleep Duration Test: t =", t_sleep, " p =", p_sleep)

'''
The p-value for intensity was 5.112e-15. Since p < 0.05, we reject the null hypothesis and 
conclude that daily activity intensity significantly affects calorie expenditure.

The p-value for sleep duration was 0.241. Since p > 0.05, we fail to reject the null hypothesis 
and conclude that sleep duration does not significantly affect calories burned.
'''





