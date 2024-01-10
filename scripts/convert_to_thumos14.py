import os
import pandas as pd
import json
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
for mode in ["train","val","test"]:
    df = pd.read_csv('/home/ubuntu/Furqan/new_dataset/annotations/20231005_ke_activity_' + mode + '_avastyle_17.csv')
    for cls in range(1,15):
        final_list = []
        frames = df[df.iloc[:,6]==cls]
        videos = frames.iloc[:,0].unique()
        for video in videos:
            frame_list = frames[frames.iloc[:,0]==video]
            intervals = ranges(frame_list.iloc[:,1].tolist())
            for interval in intervals:
                start,stop = interval
                final_list.append({"video":video,"start":start,"stop":stop})
        path = os.path.join("/home/ubuntu/Furqan/new_dataset/thumos14_annotations/",mode)
        os.makedirs(path,exist_ok=True)
        with open(path +"/" +str(cls) + ".json", "w", encoding='utf-8') as output:
            json.dump(final_list, output, ensure_ascii=False, indent=4)
    