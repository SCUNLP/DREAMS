import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import openai
from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]


def call_embedding(prompt):
    client = openai.OpenAI()
    response = client.embeddings.create(
        model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'), input=prompt,
    )

    return response


def get_exist_item_set():
    exist_item_set = set()
    for file in os.listdir(save_dir):
        user_id = os.path.splitext(file)[0]
        exist_item_set.add(user_id)
    return exist_item_set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', required=True, choices=['redial', 'opendialkg'])
    args = parser.parse_args()

    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    if not os.getenv('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY before generating embeddings.')
    batch_size = args.batch_size
    dataset = args.dataset

    save_dir = REPO_ROOT / 'save' / 'embed' / 'item' / dataset
    os.makedirs(save_dir, exist_ok=True)

    with open(REPO_ROOT / 'data' / dataset / 'id2info.json', encoding='utf-8') as f:
        id2info = json.load(f)

    # redial
    if dataset == 'redial':
        info_list = list(id2info.values())
        item_texts = []
        for info in info_list:
            item_text_list = [
                f"Title: {info['name']}", f"Genre: {', '.join(info['genre']).lower()}",
                f"Star: {', '.join(info['star'])}",
                f"Director: {', '.join(info['director'])}", f"Plot: {info['plot']}"
            ]
            item_text = '; '.join(item_text_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'star', 'director']

    # opendialkg
    if dataset == 'opendialkg':
        # item_texts = []
        # for info_dict in id2info.values():
        #     item_attr_list = [f'Name: {info_dict["name"]}']
        #     for attr, value_list in info_dict.items():
        #         if attr != 'title':
        #             item_attr_list.append(f'{attr.capitalize()}: ' + ', '.join(value_list))
        #     item_text = '; '.join(item_attr_list)
        #     item_texts.append(item_text)
        info_list = list(id2info.values())
        item_texts = []
        for info in info_list:
            if not all(key in info for key in ['name', 'genre', 'actor', 'director']):
                continue
            item_text_list = [
                f"Title: {info['name']}", f"Genre: {', '.join(info['genre']).lower()}",
                f"Actor: {', '.join(info['actor'])}",
                f"Director: {', '.join(info['director'])}"
            ]
            item_text = '; '.join(item_text_list)
            item_texts.append(item_text)
        attr_list = ['genre', 'actor', 'director']

    id2text = {}
    for item_id, info_dict in id2info.items():
        attr_str_list = [f'Title: {info_dict["name"]}']
        for attr in attr_list:
            if attr not in info_dict:
                continue
            if isinstance(info_dict[attr], list):
                value_str = ', '.join(info_dict[attr])
            else:
                value_str = info_dict[attr]
            attr_str_list.append(f'{attr.capitalize()}: {value_str}')
        item_text = '; '.join(attr_str_list)
        id2text[item_id] = item_text

    item_ids = set(id2info.keys()) - get_exist_item_set()
    while len(item_ids) > 0:
        logger.info(len(item_ids))

        # redial
        if dataset == 'redial':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]

        # opendialkg
        if dataset == 'opendialkg':
            batch_item_ids = random.sample(tuple(item_ids), min(batch_size, len(item_ids)))
            batch_texts = [id2text[item_id] for item_id in batch_item_ids]
        # import pdb; pdb.set_trace()
        response = call_embedding(batch_texts)
        batch_embeds = response.data
        for embed in batch_embeds:
            item_id = batch_item_ids[embed.index]
            with open(f'{save_dir}/{item_id}.json', 'w', encoding='utf-8') as f:
                json.dump(embed.embedding, f, ensure_ascii=False)

        item_ids -= get_exist_item_set()
