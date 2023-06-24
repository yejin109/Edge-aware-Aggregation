import pandas as pd
import collections


def load_file(data_dir, dataset_name, fname, sep):
    with open(f"{data_dir}/{dataset_name}.{fname}", 'r') as f:
        res = f.readlines()
    res = [i.strip('\n').split(sep) for i in res]
    return res


def file_to_df(raw, columns):
    return pd.DataFrame(raw, columns=columns)


def parsing_dblp_v6(lines: list, sep="\sep"):    
    indicators = ['#citation', '#arnetid', '#index', '#conf', '#year', '#@', '#!', "#*", '#%']
    keys = ['citation number', 'pid', 'index' ,'conference', 'year', 'authors', 'abstract', 'title', 'reference']
    ind2key = {ind: k for ind, k in zip(indicators, keys)}

    def set_sep(line: str):
        if line == "":
            return sep
        else:
            return line

    def set_token(line: str):
        for ind in indicators:
            line = line.replace(ind, f"<{ind2key[ind]}>")
        return line

    
    # tmp = list(map(lambda x: set_sep(x.strip('\n')), lines))
    # papers = "".join(tmp).split(sep)

    # cnt = 0
    # for i, paper in enumerate(papers):    
    #     if '#%' in paper:
    #         print(i, paper)
    #         if cnt > 5:
    #             break
    #         cnt += 1
    # # cannot recognize title token
    # papers = list(map(lambda x: set_token(x), papers))
    
    # papers = list(map(lambda x: x.split(key_token), papers))

    # papers = pd.DataFrame(papers)
    papers = collections.defaultdict(dict)
    cursor = 0
    
    for line in lines:
        line = line.strip('\n')
        if line == '':
            cursor += 1

        for ind in indicators:            
            if line.startswith(ind):
                if ind != "#%":
                    papers[cursor][ind2key[ind]] = line.lstrip(ind)
                else:
                    if ind2key[ind] not in papers[cursor].keys():
                        papers[cursor][ind2key[ind]] = line.lstrip(ind)+ ","
                    else:
                        papers[cursor][ind2key[ind]] += line.lstrip(ind)+ ","
                break

    return pd.DataFrame.from_dict(papers).T