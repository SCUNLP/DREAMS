import os
import json
import openai
import time
import random
import warnings
import csv
import copy
import difflib
from pathlib import Path
from src.model.chatgpt_mcts_dual import CHATGPT
from src.cli import create_parser, create_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
GPT4_URL = os.getenv("OPENAI_BASE_URL", "").strip()
GPT4 = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
REQUEST_DELAY = float(os.getenv("DREAMS_REQUEST_DELAY", "0"))

pool_keys = []
available = []


def configure_api(cli_key=None):
    global pool_keys, available, GPT4_URL
    if cli_key:
        os.environ["OPENAI_API_KEY"] = cli_key
    GPT4_URL = os.getenv("OPENAI_BASE_URL", "").strip()
    pool_keys = [key.strip() for key in os.getenv("DREAMS_OPENAI_API_KEYS", "").split(",") if key.strip()]
    primary_key = os.getenv("OPENAI_API_KEY", "").strip()
    if primary_key and primary_key not in pool_keys:
        pool_keys.insert(0, primary_key)
    if not pool_keys:
        raise RuntimeError("Set OPENAI_API_KEY before running the simulator.")
    available = [1] * len(pool_keys)


def openai_client(api_key):
    kwargs = {"api_key": api_key}
    if GPT4_URL:
        kwargs["base_url"] = GPT4_URL
    return openai.OpenAI(**kwargs)


def maybe_delay():
    if REQUEST_DELAY > 0:
        time.sleep(REQUEST_DELAY)

def get_exist_dialog_set():
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0].split("_p")[0]
        exist_id_set.add(file_id)
    return exist_id_set


def get_seeker_feelings(seeker_instruct, seeker_prompt, mode):
    seeker_prompt += '''
            The Seeker notes how he feels to himself in one sentence.

            What aspects of the recommended movies meet your preferences? What aspects of the recommended movies may not meet your preferences? What do you think of the performance of this recommender?
            What would the Seeker think to himself? What would his internal monologue be?
            The response should be short (as most internal thinking is short) and strictly follow your Seeker persona .
            Do not include any other text than the Seeker's thoughts.
            Respond in the first person voice (use "I" instead of "Seeker") and speaking style of Seeker. Pretend to be Seeker!
                '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text

    elif mode == 'gpt':
        print(seeker_instruct)
        print("=======")
        print(seeker_prompt)

        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]

        for key_ind in range(len(pool_keys)):
            key = pool_keys[key_ind]
            if available[key_ind] == 1:
                client = openai_client(key)

                try:
                    response = client.chat.completions.create(
                        model='gpt-4o-mini', messages=messages, temperature=0, seed=0,
                    ).choices[0].message.content
                    # print('1')
                    print(response)
                    maybe_delay()
                    break
                except Exception as e:
                    print("failed key index=", key_ind)
                    print(e)
                    available[key_ind] = 0
        if sum(available) == 0:
            raise RuntimeError
        return response


def get_seeker_insights(seeker_instruct, seeker_prompt, seeker_feelings, mode):
    seeker_prompt += '''

                Here is your feelings about the last sentence:
                '''
    seeker_prompt += seeker_feelings
    seeker_prompt += '''
            What's the insight and next plan of the Seeker think to himself in one sentence based on the information above?

            What would the Seeker think to himself? What would his internal monologue be?
            The response should be short (as most internal thinking is short).
            Do not include any other text than the Seeker's thoughts.
            Respond in the first person voice (use "I" instead of "Seeker") and speaking style of Seeker. Pretend to be Seeker!
                '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':

        client = openai_client(pool_keys[0])
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        response = client.chat.completions.create(
            model='gpt-4o-mini', messages=messages, temperature=0, seed=0,
        ).choices[0].message.content

        print(response)

        return response


def get_seeker_text(seeker_instruct, seeker_prompt, seeker_feelings, seeker_insights, mode):
    seeker_prompt += '''

        Here is your feelings about the last sentence:
        '''
    seeker_prompt += seeker_feelings
    seeker_prompt += '''
    Here is your insights about what to do next:
    '''
    seeker_prompt += seeker_insights
    seeker_prompt += '''
What does the Seeker says next.

Keep your response brief. Use casual language and vary your wording.
Make sure your response matches your Seeker persona, your preferred attributes, and your conversation context.
Do not include your feelings into the response to the Seeker!
Respond in the first person voice (use "I" instead of "Seeker", use "you" instead of "recommender") and speaking style of the Seeker. Vary your wording!
    '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        client = openai_client(pool_keys[0])
        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        response = client.chat.completions.create(
            model='gpt-4o-mini', messages=messages, temperature=0, seed=0,
        ).choices[0].message.content

        print(response)

        return response


# TODO
def get_seeker_text_2(seeker_instruct, seeker_prompt, seeker_feelings, mode):

    # TODO
    seeker_prompt += '''
Pretend to be the Seeker! What do you say next.

Keep your response brief. Use casual language and vary your wording.
Make sure your response matches your Seeker persona, your preferred attributes, and your conversation context.
Do not include your feelings into the response to the Seeker!
Respond in the first person voice (use "I" instead of "Seeker", use "you" instead of "recommender") and speaking style of the Seeker.
    '''

    if mode == 'llama':

        llama_prompt = seeker_instruct + seeker_prompt
        output = llama.generate(llama_prompt, sampling_params)

        return output[0].outputs[0].text
    elif mode == 'gpt':
        print(seeker_instruct)
        print("====")
        print(seeker_prompt)

        messages = [{'role': 'system', 'content': seeker_instruct}, {'role': 'user', 'content': seeker_prompt}]
        for key_ind in range(len(pool_keys)):
            key = pool_keys[key_ind]
            if available[key_ind] == 1:
                client = openai_client(key)
                try:
                    response = client.chat.completions.create(
                        model='gpt-4o-mini', messages=messages, temperature=0, seed=0,
                    ).choices[0].message.content
                    print('3')
                    print(response)
                    maybe_delay()
                    break
                except Exception as e:
                    print("failed key index=", key_ind)
                    print(e)
                    available[key_ind] = 0
        if sum(available) == 0:
            raise RuntimeError

        return response


def get_instruction(dataset):
    if dataset.startswith('redial'):
        item_with_year = True
    elif dataset.startswith('opendialkg'):
        item_with_year = False
    # TODO
    if item_with_year is True:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        seeker_instruction_template = '''You are a seeker chatting with a recommender for movie recommendation.
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.
Your preferred movie should cover at least one of these stars: {}.
Your preferred movie should cover at least one of these directors: {}.
You must follow the instructions below during chat.
1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie. However, do not continuously ask for particularly detailed content like "memorable scenes".
2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
4. Once you are sure that the recommended movie exactly covers all your preferred genres, and covers at least one of the actors and directors you like, you should accept it and end the conversation with a special token "[END]" at the end of your response.
5. If the recommender asks your preferred genre, you should describe your preferred movie genre in your own words and you'd better not clearly mention the movie type.
6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent.
7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!
8. If the recommender asks you about your preferred genre, actors, or directors, you should always answer it.
9. In a single conversation, only reveal one of your preferred genres, actors, or directors.
10. If the recommender tells you that the star and director you like have never collaborated, try telling them about other stars or directors you like.
'''
    else:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''

        seeker_instruction_template = '''You are a seeker chatting with a recommender for movie recommendation.
Your Seeker persona: {}
Your preferred movie should cover those genres at the same time: {}.
Your preferred movie should cover at least one of these stars: {}.
Your preferred movie should cover at least one of these directors: {}.
You must follow the instructions below during chat.
1. If the recommender recommends movies to you, you should always ask the detailed information about the each recommended movie.
2. Pretend you have little knowledge about the recommended movies, and the only information source about the movie is the recommender.
3. After getting knowledge about the recommended movie, you can decide whether to accept the recommendation based on your preference.
4. Once you are sure that the recommended movie exactly covers all your preferred genres, and covers at least one of the actors and directors you like, you should accept it and end the conversation with a special token "[END]" at the end of your response.
5. If the recommender asks your preferred genre, you should describe your preferred movie genre in your own words and you'd better not clearly mention the movie type.
6. You can chit-chat with the recommender to make the conversation more natural, brief, and fluent.
7. Your utterances need to strictly follow your Seeker persona. Vary your wording and avoid repeating yourself verbatim!
8. If the recommender asks you about your preferred genre, actors, or directors, you should always answer it.
9. In a single conversation, only reveal one of your preferred genres, actors, or directors.
10. If the recommender tells you that the star and director you like have never collaborated, try telling them about other stars or directors you like.
'''

    return recommender_instruction, seeker_instruction_template


if __name__ == '__main__':
    parser = create_parser('Run DREAMS with the LLM user simulator')
    args = parser.parse_args()
    configure_api(args.api_key)
    if args.ignore_warnings:
        warnings.filterwarnings("ignore")
    trace = create_trace(args, REPO_ROOT)

    save_dir = REPO_ROOT / f'save_{args.turn_num}' / 'chat' / args.crs_model / args.dataset
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)

    llama = None
    sampling_params = None

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset)

    with open(REPO_ROOT / 'data' / args.kg_dataset / 'entity2id.json', 'r', encoding="utf-8") as f:
        entity2id = json.load(f)

    with open(REPO_ROOT / 'data' / args.kg_dataset / 'id2info.json', 'r', encoding="utf-8") as f:
        id2info = json.load(f)

    profiles = []
    with open(REPO_ROOT / 'data' / 'profile.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            profiles.append(row[3])

    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())

    dialog_id2data = {}
    with open(REPO_ROOT / 'data' / args.dataset / 'test_data_processed.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line

    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dialog_set()

    attribute_list = ['action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama',
                      'family', 'fantasy', 'film-noir', 'game-show', 'history', 'horror', 'music', 'musical',
                      'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller',
                      'war', 'western']
    chatgpt_paraphrased_attribute = {'action': 'thrilling and adrenaline-pumping action movie',
                                     'adventure': 'exciting and daring adventure movie',
                                     'animation': 'playful and imaginative animation',
                                     'biography': 'inspiring and informative biography',
                                     'comedy': 'humorous and entertaining flick',
                                     'crime': 'suspenseful and intense criminal film',
                                     'documentary': 'informative and educational documentary',
                                     'drama': 'emotional and thought-provoking drama',
                                     'family': 'heartwarming and wholesome family movie',
                                     'fantasy': 'magical and enchanting fantasy movie',
                                     'film-noir': 'dark and moody film-noir',
                                     'game-show': 'entertaining and interactive game-show',
                                     'history': 'informative and enlightening history movie',
                                     'horror': 'chilling, terrifying and suspenseful horror movie',
                                     'music': 'melodious and entertaining music',
                                     'musical': 'theatrical and entertaining musical',
                                     'mystery': 'intriguing and suspenseful mystery',
                                     'news': 'informative and current news',
                                     'reality-tv': 'dramatic entertainment and reality-tv',
                                     'romance': 'romantic and heartwarming romance movie with love story',
                                     'sci-fi': 'futuristic and imaginative sci-fi with futuristic adventure',
                                     'short': 'concise and impactful film with short story',
                                     'sport': 'inspiring and motivational sport movie',
                                     'talk-show': 'informative and entertaining talk-show such as conversational program',
                                     'thriller': 'suspenseful and thrilling thriller with gripping suspense',
                                     'war': 'intense and emotional war movie and wartime drama',
                                     'western': 'rugged and adventurous western movie and frontier tale'}

    attribute_candidates = [['comedy', 'drama', 'romance'],
                            ['adventure', 'animation', 'comedy'],
                            ['action', 'adventure', 'sci-fi'],
                            ['action', 'crime', 'drama'],
                            ['action', 'adventure', 'comedy'],
                            ['action', 'comedy', 'crime'],
                            ['action', 'crime', 'thriller'],
                            ['crime', 'drama', 'thriller'],
                            ['action', 'adventure', 'fantasy'],
                            ['horror', 'mystery', 'thriller'],
                            ['action', 'adventure', 'drama'],
                            ['crime', 'drama', 'mystery'],
                            ['action', 'adventure', 'animation'],
                            ['adventure', 'comedy', 'family'],
                            ['action', 'adventure', 'thriller'],
                            ['comedy', 'drama', 'family'],
                            ['drama', 'horror', 'mystery'],
                            ['biography', 'drama', 'history'],
                            ['biography', 'crime', 'drama'],
                            ]
    star_candidates = [['Michael Douglas', 'Shirley MacLaine', 'Julia Roberts'],
                       ['John Goodman', 'Eddie Murphy', 'Phil Harris'],
                       ['Robert Downey Jr.', 'Jennifer Lawrence', 'Chris Evans'],
                       ['Christian Bale', 'Gene Hackman', 'Denzel Washington'],
                       ['Jackie Chan', 'Tommy Lee Jones', 'Will Smith'],
                       ['Martin Lawrence', 'Jackie Chan', 'Eddie Murphy'],
                       ['Jason Statham', 'Mel Gibson', 'Danny Glover'],
                       ['Anthony Hopkins', 'Kevin Costner', 'Robert De Niro'],
                       ['Mark Hamill', 'Johnny Depp', 'Natalie Portman'],
                       ['Patrick Wilson', 'Vera Farmiga', 'Rose Byrne'],
                       ['Russell Crowe', 'Leonardo DiCaprio', 'Viggo Mortensen'],
                       ['Denzel Washington', 'Julia Roberts', 'Morgan Freeman'],
                       ['Veronica Taylor', 'Jack Black', 'Akemi Okamura'],
                       ['Robin Williams', 'Ben Stiller', 'Rick Moranis'],
                       ['Sylvester Stallone', 'Sean Connery', 'Daniel Craig'],
                       ['Vanessa Hudgens', 'Zac Efron', 'Ashley Tisdale'],
                       ['Jesse Bradford', 'Rod Taylor', 'Tippi Hedren'],
                       ['Chiwetel Ejiofor', 'Patrick McGoohan', 'Ralph Fiennes'],
                       ['Al Pacino', 'Johnny Depp', 'Hilary Swank']
    ]

    director_candidates = [['Rob Reiner', 'Woody Allen'],
                       ['Carlos Saldanha', 'Wolfgang Reitherman'],
                       ['Michael Bay', 'Roland Emmerich'],
                       ['F. Gary Gray'],
                       ['Barry Sonnenfeld'],
                       ['Brett Ratner', 'Michael Bay'],
                       ['Richard Donner', 'Robert Rodriguez'],
                       ['Ethan Coen', 'Joel Coen'],
                       ['George Lucas', 'Gore Verbinski'],
                       ['James Wan'],
                       ['Peter Jackson', 'Ridley Scott'],
                       ['Tony Scott', 'David Fincher'],
                       ['Kunihiko Yuyama', 'Mamoru Hosoda'],
                       ['Shawn Levy', 'Brian Levant'],
                       ['Sylvester Stallone', 'Scott Waugh'],
                       ['Kenny Ortega'],
                       ['Mike Flanagan'],
                       ['Steven Spielberg', 'Mel Gibson'],
                       ['Sidney Lumet', 'Martin Scorsese']
    ]
    for profile_ind in range(12, len(profiles)): #len(profiles)
        profile_str = profiles[profile_ind]
        for attribute_ind in range(0, len(attribute_candidates)): #len(attribute_candidates)
            preferred_attribute_list = attribute_candidates[attribute_ind]
            preferred_actor_list = star_candidates[attribute_ind]
            preferred_director_list = director_candidates[attribute_ind]
            # while len(dialog_id_set) > 0:

            print(len(dialog_id_set))
            random.seed(len(dialog_id_set) + args.seed)
            dialog_id = random.choice(tuple(dialog_id_set))

            data = dialog_id2data[dialog_id]
            conv_dict = copy.deepcopy(data)  # for model
            # context = conv_dict['context']
            context = ['Hello']
            conv_dict['context'] = context


            target_list = []
            for k, v in id2info.items():
                if set(v['genre']) == set(preferred_attribute_list):
                    if any(actor in v.get('star', v.get('actor', [])) for actor in preferred_actor_list):
                        if any(director in v['director'] for director in preferred_director_list):
                            target_list.append(v['name'])

            if len(target_list) == 0:
                print("empty target list")
                continue

            # TODO
            preferred_attribute_str = ', '.join([chatgpt_paraphrased_attribute.get(i) for i in preferred_attribute_list]) #

            seeker_instruct = seeker_instruction_template.format(profile_str, preferred_attribute_str, preferred_actor_list, preferred_director_list)
            seeker_prompt = '''
            Conversation History
            #############
            '''
            context_dict = []  # for save

            for i, text in enumerate(context):
                if len(text) == 0:
                    continue
                if i % 2 == 0:
                    role_str = 'user'
                    seeker_prompt += f'Seeker: {text}\n'
                else:
                    role_str = 'assistant'
                    seeker_prompt += f'Recommender: {text}\n'
                context_dict.append({
                    'role': role_str,
                    'content': text
                })

            rec_success = False
            rec_success_rec = False
            rec_success_rec_1 = False
            rec_success_rec_5 = False
            rec_success_rec_10 = False
            recall_1 = False
            recall_5 = False
            recall_10 = False
            recommendation_template = "I would recommend the following items: {}:"
            recommender = CHATGPT(
                seed=args.seed,
                debug=args.debug,
                kg_dataset=args.kg_dataset,
                mcts_iterations=args.mcts_iterations,
                simulation_workers=args.simulation_workers,
                trace=trace,
                repo_root=REPO_ROOT,
            )
            recommender.conversation_state = {
                        'user_preferences': {'genres': [], 'actors': [], 'directors': []},
                        'recommended_items': [],
                        'actions_taken': [],
                        'context': [],
                        'user_attitude': 'undecided',
                        'turn_count': 0
                    }

            for i in range(0, args.turn_num):
                best_action = recommender.select_action(conv_dict)
            # print(conv_dict)
                rec_items, conv_str = recommender.get_rec(conv_dict)
                rec_labels = []
                rerank_item = ""
                for item in target_list:
                    if item in entity2id.keys():
                        rec_labels.append(entity2id[item])

                for rec_label in rec_labels:
                    if rec_label == rec_items[0][0]:
                        rec_success_rec_1 = True
                        break
                    else:
                        rec_success_rec_1 = False

                for rec_label in rec_labels:
                    if rec_label in rec_items[0][:5]:
                        rec_success_rec_5 = True
                        break
                    else:
                        rec_success_rec_5 = False

                for rec_label in rec_labels:
                    if rec_label in rec_items[0][:10]:
                        rec_success_rec_10 = True
                        break
                    else:
                        rec_success_rec_10 = False

                cand_list = []
                refined_query = ""
                recall_1 = False
                recall_5 = False
                recall_10 = False

                if best_action == "ItemRecommendation":
                    context = conv_dict['context']
                    context_list = []  # for model

                    for i, text in enumerate(context):
                        if not isinstance(text, str):
                            text = str(text)
                        if len(text) == 0:
                            continue
                        if i % 2 == 0:
                            role_str = 'user'
                        else:
                            role_str = 'assistant'
                        context_list.append({
                            'role': role_str,
                            'content': text
                        })

                    conv_str = ""

                    for context in context_list:
                        conv_str += f"{context['role']}: {context['content']} "

                    cand_list, _, recommender_text, refined_query = recommender.execute_action(best_action, conv_dict)
                    for rec_label in rec_labels:
                        if rec_label == cand_list[0][0]:
                            recall_1 = True
                            break
                        else:
                            recall_1 = False

                    for rec_label in rec_labels:
                        if rec_label in cand_list[0][:5]:
                            recall_5 = True
                            break
                        else:
                            recall_5 = False

                    for rec_label in rec_labels:
                        if rec_label in cand_list[0][:10]:
                            recall_10 = True
                            break
                        else:
                            recall_10 = False

                else:
                    result = recommender.execute_action(best_action, conv_dict)
                    if len(result) == 2:
                        _, recommender_text = result
                    elif len(result) == 4:
                        cand_list, _, recommender_text, refined_query = result



                if best_action == "ItemRecommendation":
                    rerank_item = _["rec_item"]
                    rec_items = cand_list

                for target in target_list:
                    similarity = difflib.SequenceMatcher(None, target.lower(), rerank_item.lower()).ratio()
                    if similarity > 0.7:
                        rec_success_rec = True
                        break

                context_dict.append({
                    'role': 'assistant',
                    'content': recommender_text,
                    # 'entity': recommender_resp_entity,
                    'rec_items': rec_items[0],
                    'best_action': best_action,
                    'rec_success_dialogue': rec_success,
                    'rec_success_rec_1': rec_success_rec_1,
                    'rec_success_rec_5': rec_success_rec_5,
                    'rec_success_rec_10': rec_success_rec_10,
                    'rec_success_rec': rec_success_rec,
                    'recall_1': recall_1,
                    'recall_5': recall_5,
                    'recall_10': recall_10,
                    'refined_query': refined_query,
                    'conv_str': conv_str
                })
                conv_dict['context'].append(recommender_text)

                seeker_prompt += f'Recommender: {recommender_text}\n'

                seeker_feelings = get_seeker_feelings(seeker_instruct, seeker_prompt, mode='gpt')

                seeker_text = get_seeker_text_2(seeker_instruct, seeker_prompt, seeker_feelings, mode='gpt')

                seeker_prompt += f'Seeker: {seeker_text}\n'


                context_dict.append({
                    'role': 'user',
                    'content': seeker_text,
                    "feelings": seeker_feelings,
                    # 'score': satisfaction_score
                })
                if trace:
                    trace.record(
                        'real_turn',
                        action=best_action,
                        assistant=recommender_text,
                        feedback=seeker_text,
                        feelings=seeker_feelings,
                    )
                    trace.render(state=recommender.conversation_state)

                conv_dict['context'].append(seeker_text)
                conv_dict['attributes'] = preferred_attribute_list
                conv_dict['profile'] = [profile_str]


                if seeker_text.find("[END]") != -1:
                    rec_success = True
                    context_dict[-2]['rec_success_dialogue'] = True
                    break

            conv_dict['context'] = context_dict
            data['simulator_dialog'] = conv_dict


            # save
            with open(f'{save_dir}/{dialog_id}_profile{str(profile_ind)}_attribute{str(attribute_ind)}.json', 'w',
                      encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            dialog_id_set -= get_exist_dialog_set()
