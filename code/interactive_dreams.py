import os
import json
from colorama import Fore, Style, init
import warnings
import random
import copy
import csv
from thefuzz import fuzz
import difflib
from pathlib import Path

from src.model.chatgpt_mcts_dual import CHATGPT
from src.cli import create_parser, create_trace

init(autoreset=True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def fuzzy_match(term, candidates, threshold=80):
    """
    Perform fuzzy matching between a term and a list of candidates.

    Args:
        term (str): The term to match
        candidates (list): List of candidate strings
        threshold (int): Minimum score threshold (0-100)

    Returns:
        bool: True if a match is found, False otherwise
    """
    term = term.lower()
    for candidate in candidates:
        candidate = candidate.lower()
        # Check exact match
        if term == candidate:
            return True
        # Check if term is a substring of candidate or vice versa
        if term in candidate or candidate in term:
            return True
        # Use fuzzy matching for more complex cases
        if fuzz.ratio(term, candidate) >= threshold:
            return True
    return False

def normalize_genre(genre):
    """
    Normalize genre names by handling common variations.

    Args:
        genre (str): The genre name to normalize

    Returns:
        str: Normalized genre name
    """
    # Common variations mapping
    variations = {
        'rom': 'romance',
        'romcom': 'comedy romance',
        'romantic comedy': 'comedy romance',
        'sci-fi': 'science fiction',
        'scifi': 'science fiction',
        'historical': 'history',
        'fantasy': 'fantasy',
        'criminal': 'crime',
        'war film': 'war',
        'documentary film': 'documentary',
        'horror movie': 'horror',
        'thriller movie': 'thriller',
        'action movie': 'action',
    }

    genre = genre.lower()

    for key, value in variations.items():
        if fuzzy_match(genre, [key], threshold=85):
            return value

    return genre


def get_exist_dialog_set():
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0].split("_p")[0]
        exist_id_set.add(file_id)
    return exist_id_set

if __name__ == '__main__':
    parser = create_parser('Run DREAMS with a human user')
    args = parser.parse_args()
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    if args.ignore_warnings:
        warnings.filterwarnings("ignore")
    trace = create_trace(args, REPO_ROOT)

    save_dir = REPO_ROOT / f'save_human_{args.turn_num}' / 'chat' / args.crs_model / args.dataset
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)

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

    # TODO
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
    for attribute_ind in range(14, len(attribute_candidates)):
        preferred_attribute_list = attribute_candidates[attribute_ind]
        preferred_actor_list = star_candidates[attribute_ind]
        preferred_director_list = director_candidates[attribute_ind]

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
            if not all(key in v for key in ['genre', 'star', 'director']):
                continue

            genre_match = all(any(fuzzy_match(preferred_genre, [normalize_genre(movie_genre)])
                                  for movie_genre in v['genre'])
                              for preferred_genre in preferred_attribute_list)

            if genre_match:
                if set(v.get('star', v.get('actor', []))) & set(preferred_actor_list):
                    if set(v['director']) & set(preferred_director_list):
                        target_list.append(v['name'])

        if len(target_list) == 0:
            print("empty target list")
            continue

        # print('preferred_genre:', preferred_attribute_list)
        # print('preferred_actor:', preferred_actor_list)
        # print('preferred_director:', preferred_director_list)

        preferred_attribute_str = ', '.join([chatgpt_paraphrased_attribute.get(i) for i in preferred_attribute_list])
        # print(preferred_attribute_str)

        context_dict = []  # for save

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_dict.append({
                'role': role_str,
                'content': text
            })

        rec_success = False
        rec_success_rec = False
        rec_success_rec_1 = False
        rec_success_rec_5 = False
        rec_success_rec_10 = False
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
                # import pdb; pdb.set_trace()

                # for rec_label in rec_labels:
                #     for j in range(len(rec_items[0])):
                #         if rec_label == rec_items[0][j]:
                #             rec_success_rec = True
                #             break
                #     else:
                #         rec_success_rec = False
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

            print("recommender: ")
            print(recommender_text)
            conv_dict['context'].append(recommender_text)

            seeker_text = input(f"{Fore.BLUE}User: {Style.RESET_ALL}")

            if seeker_text.lower() == '/exit':
                print(f"{Fore.YELLOW}Exiting demo. Goodbye!{Style.RESET_ALL}")
                break
            if seeker_text.lower() == '/prefs':
                print(recommender.conversation_state)



            context_dict.append({
                    'role': 'user',
                    'content': seeker_text,
            })
            if trace:
                trace.record(
                    'real_turn',
                    action=best_action,
                    assistant=recommender_text,
                    feedback=seeker_text,
                )
                trace.render(state=recommender.conversation_state)

            conv_dict['context'].append(seeker_text)
            conv_dict['attributes'] = preferred_attribute_list

            if seeker_text.find("[END]") != -1:
                rec_success = True
                context_dict[-2]['rec_success_dialogue'] = True
                break

        conv_dict['context'] = context_dict
        data['simulator_dialog'] = conv_dict

        with open(f'{save_dir}/{dialog_id}_attribute{str(attribute_ind)}.json', 'w',
                      encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        dialog_id_set -= get_exist_dialog_set()
