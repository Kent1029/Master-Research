import pandas as pd

data = pd.read_csv("deal_youtube_c40_Eyes_Blink_Counter.csv")
blink_count = data['blink_count'].values
avg_blink_duration = data['avg_blink_duration'].values

print("blink_count::", blink_count)
print("type(blink_count)",type(blink_count))
#print("avg_blink_duration::", avg_blink_duration)
