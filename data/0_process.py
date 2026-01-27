'''
Similar data-preprocessing of D3, with split ratio of 4:4:2
'''
import fire
from loguru import logger
import json
from tqdm import tqdm
import random
import time
import datetime
import csv
import os
import numpy as np
import ipdb
    

def get_timestamp_start(year, month):
    return int(datetime.datetime(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())

def gao(category, metadata=None, reviews=None, K=5, st_year=2017, st_month=10, ed_year=2018, ed_month=11, output=True):
    if st_year < 1996:
        return
    start_timestamp = get_timestamp_start(st_year, st_month)
    end_timestamp = get_timestamp_start(ed_year, ed_month)
    logger.info(f"from {start_timestamp} to {end_timestamp}")
    if metadata is None:
        with open(f'../raw_data/meta_{category}.json') as f:
            metadata = [json.loads(line) for line in f]
        try:
            with open(f'../raw_data/{category}_5.json') as f:
                reviews = [json.loads(line) for line in f]
        except:
            with open(f'../raw_data/{category}.json') as f:
                reviews = [json.loads(line) for line in f]
    else:
        metadata = metadata
        reviews = reviews
    logger.info(f"from {category} metadata: {len(metadata)} reviews: {len(reviews)}")
    
    # item: asin, title, price, imUrl, related, salesRank, categories, description
    users = set()
    items = set()
    for review in tqdm(reviews):
        if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
            continue        
        users.add(review['reviewerID'])
        items.add(review['asin'])
    # print(len(users), len(items))

    logger.info(f"users: {len(users)}, items: {len(items)}, reviews: {len(reviews)}, density: {len(reviews) / (len(users) * len(items))}")
    remove_users = set()
    remove_items = set()
    

    id_title = {}
    for meta in tqdm(metadata):
        if ('title' not in meta) or (meta['title'].find('<span id') > -1):
            remove_items.add(meta['asin'])
            continue
        meta['title'] = meta["title"].replace("&quot;", "\"").replace("&amp;", "&").strip(" ").strip("\"")
        if len(meta['title']) > 1 and len(meta['title'].split(" ")) <= 20: # remove the item without title # remove too long title
            id_title[meta['asin']] = meta['title']
        else:
            remove_items.add(meta['asin'])
    for review in tqdm(reviews):
        if review['asin'] not in id_title:
            remove_items.add(review['asin'])

    while True:
        new_reviews = []
        flag = False
        total = 0
        users = dict()
        items = dict()
        new_reviews = []
        for review in tqdm(reviews):
            if int(review["unixReviewTime"]) < start_timestamp or int(review["unixReviewTime"]) > end_timestamp:
                continue
            if review['reviewerID'] in remove_users or review['asin'] in remove_items:
                continue          
            if review['reviewerID'] not in users:
                users[review['reviewerID']] = 0
            users[review['reviewerID']] += 1
            if review['asin'] not in items:
                items[review['asin']] = 0
                
            items[review['asin']] += 1
            total += 1
            new_reviews.append(review)
            
        for user in users:
            if users[user] < K:
                remove_users.add(user)
                flag = True
        
        for item in items:
            if items[item] < K:
                remove_items.add(item)
                flag = True

        logger.info(f"users: {len(users)}, items: {len(items)}, reviews: {total}, density: {total / (len(users) * len(items))}")
        if st_year > 1996 and len(items) < 10000:
            break
        if not flag:
            break

    if st_year > 1996 and len(items) < 10000:
        gao(category, metadata = metadata, reviews = reviews, K=K, st_year=st_year - 1, st_month=st_month, ed_year=ed_year, ed_month=ed_month, output=True)
        return
    
    # save meta
    meta_data = {}
    for meta in tqdm(metadata):
        if meta['asin'] in items.keys(): # put the valid items into temp meta file
            meta_data[meta['asin']] = meta
            
    np.save("meta_temp.npy", meta_data)
    logger.info(f"{len(items)-len(meta_data)} items have NO meta info")
    logger.info(f"remove_users: {len(remove_users)}, remove_items: {len(remove_items)}")
    
    reviews = new_reviews

    if not output:
        return 
    # get all the review data

    interact = dict()
    for review in tqdm(new_reviews):
        user = review['reviewerID']
        item = review['asin']
        if user not in interact:
            interact[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'reviews': []
            }
        interact[user]['items'].append(item)
        interact[user]['ratings'].append(review['overall'])
        interact[user]['timestamps'].append(review['unixReviewTime'])
    
    # shuffle items and assign the id to each item
    items = list(items.keys())
    random.seed(42)
    random.shuffle(items)

    # create item map
    item2id = dict()
    id2item = dict()
    count = 0
    for item in items:
        item2id[item] = count
        id2item[count] = item  # convert into string to meet the json requirement
        count += 1
    item_mappings = {"item2id": item2id,"id2item": id2item}

    # create user map
    user_list = list(interact.keys())
    random.shuffle(user_list)
    user2id = dict()
    id2user = dict()
    count = 0
    for user in user_list:
        user2id[user] = count
        id2user[count] = user  # convert into string to meet the json requirement
        count += 1
    user_mappings = {"user2id": user2id,"id2user": id2user}

    interaction_list = []   
    for key in tqdm(interact.keys()): # for each user
        items = interact[key]['items']
        ratings = interact[key]['ratings']
        timestamps = interact[key]['timestamps']
        all = list(zip(items, ratings, timestamps))
        res = sorted(all, key=lambda x: int(x[2]))
        items, ratings, timestamps = zip(*res)
        items, ratings, timestamps = list(items), list(ratings), list(timestamps)
        interact[key]['items'] = items
        interact[key]['ratings'] = ratings
        interact[key]['timestamps'] = timestamps
        interact[key]['item_ids'] = [item2id[item] for item in items]
        interact[key]['title'] = [id_title[item] for item in items]
        for i in range(1, len(items)):
            st = max(i - 10, 0)
            interaction_list.append([key, interact[key]['items'][st:i], interact[key]['items'][i], interact[key]['item_ids'][st:i], interact[key]['item_ids'][i], interact[key]['title'][st:i], interact[key]['title'][i], interact[key]['ratings'][st:i], interact[key]['ratings'][i], interact[key]['timestamps'][st:i], interact[key]['timestamps'][i]])
    logger.info(f"interaction_list: {len(interaction_list)}")


    # split interaction_list into 5 equal parts: D0–D4
    interaction_list = sorted(interaction_list, key=lambda x: int(x[-1]))

    n = len(interaction_list)
    split_size = n // 5
    print("Each Period Interaction Num:", split_size)

    # create directories
    for i in range(5):
        dir_name = f"./D{i}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    # write splits
    for i in range(5):
        start = i * split_size
        end = (i + 1) * split_size if i < 4 else n

        with open(
            f"./D{i}/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}.csv",
            "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                'user_id', 'item_asins', 'item_asin',
                'history_item_id', 'item_id',
                'history_item_title', 'item_title',
                'history_rating', 'rating',
                'history_timestamp', 'timestamp'
            ])
            writer.writerows(interaction_list[start:end])

        logger.info(f"D{i} {category}: {end - start}")
    
    for i in range(5):
        start = i * split_size
        end = (i + 1) * split_size if i < 4 else n
        logger.info(f"D{i} {category}: {end - start}")

    # read and save meta
    title_map = {'recid2title':{},'item2title':{}}
    description_map = {'recid2description':{}, 'item2description':{}}
    category_map = {'recid2category':{}, 'item2category':{}}
    brand_map = {'recid2brand':{}, 'item2brand':{}}
    cnt_t, cnt_d, cnt_c, cnt_b = 0, 0, 0, 0
    for asin, i_id in item2id.items():
        each = meta_data[asin]  
        # item: asin, title, price, imUrl, related, salesRank, categories, description
        title, description, catts, brand = each.get("title", None), each.get("description", None), each.get("category", None), each.get("brand", None)
        
        # 1. title
        title = f"title_{i_id}" if title is None else title
        title_map['recid2title'][i_id] = title
        title_map['item2title'][asin] = title
        if title[:6] == "title_":
            cnt_t += 1

        # 2. description
        description = "unknown" if (description is None or len(description)==0) else description[0]
        description_map['recid2description'][i_id] = "unknown" if description is None else description
        description_map['item2description'][asin] = "unknown" if description is None else description
        if description == "unknown":
            cnt_d += 1

        # 3. category
        if catts is None:
            cats = [f'{category}']
        else:
            cats = catts
        if '</span></span></span>' in cats:
            cats.remove('</span></span></span>')
        for idx, cat in enumerate(cats):
            if "&amp" in cat:
                temp = cat.split(' &amp')
                cats[idx] = ''.join(temp)
        category_map['recid2category'][i_id] = cats
        category_map['item2category'][asin] = cats

        # 4. brand
        brand = "unknown" if brand is None else brand
        brand_map['recid2brand'][i_id] = brand
        brand_map['item2brand'][asin] = brand
        if brand == "unknown":
            cnt_b += 1

    # title and category with prompt: "The item title is {title}, and the category is {category}"
    def get_prompt(ttl, desc, cat, brand):
        cats = ' , '.join(cat)
        # ipdb.set_trace()
        return f"The item title is {ttl}. {desc}. It has the categories of {cats}. The brand of this item is {brand}"

    comb = {"recid2combine":{}}
    t_map = title_map['recid2title']
    d_map = description_map['recid2description']
    c_map = category_map['recid2category']
    b_map = brand_map['recid2brand']
    for i_id in id2item.keys():
        i_id = int(i_id)
        comb['recid2combine'][i_id] = get_prompt(t_map[i_id], d_map[i_id], c_map[i_id], b_map[i_id])

    # save json file
    if not os.path.exists("./info"):
        os.mkdir("./info")

    # save all maps
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_item_map.npy", item_mappings)
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_user_map.npy", user_mappings)
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_title_map.npy", title_map)
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_description_map.npy", description_map)
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_category_map.npy", category_map)
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_brand_map.npy", brand_map)
    np.save(f"./info/{category}_{K}_{st_year}-{st_month}-{ed_year}-{ed_month}_combine_tdcb_maps.npy", comb) # save combine dict

    logger.info(f"{cnt_t} ({round(cnt_t/len(item2id),2)*100}%) items have no titles")
    logger.info(f"{cnt_d} ({round(cnt_d/len(item2id),2)*100}%) items have no descriptions")
    logger.info(f"{cnt_c} ({round(cnt_c/len(item2id),2)*100}%) items have no categories")
    logger.info(f"{cnt_b} ({round(cnt_b/len(item2id),2)*100}%) items have no brand")

    logger.info("Done!")
    

if __name__ == '__main__':
    fire.Fire(gao)
