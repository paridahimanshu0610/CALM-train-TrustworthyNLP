import argparse
import json
import datetime
import os

'''
{
    "orig_data": {
        "instruction": "Problem: Xiaoming bought a pen for 5 yuan and a book for 8 yuan. He now has 10 yuan left. How much money did he originally have?",
        "input": "",
        "output": "\nLet X be the amount of money Xiaoming originally had. According to the problem, we form the equation:\nX - 5 - 8 = 10\nSimplifying:\nX = 23\nTherefore, Xiaoming originally had 23 yuan."
    },
    "convert": {
        "id": "xxx",
        "conversations": [
            {
                "from": "human",
                "value": "Problem: Xiaoming bought a pen for 5 yuan and a book for 8 yuan. He now has 10 yuan left. How much money did he originally have?"
            },
            {
                "from": "assistant",
                "value": "\nLet X be the amount of money Xiaoming originally had. According to the problem, we form the equation:\nX - 5 - 8 = 10\nSimplifying:\nX = 23\nTherefore, Xiaoming originally had 23 yuan."
            }
        ]
    }
}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_data",
    )
    parser.add_argument(
        "--write_data",
    )
    parser.add_argument(
        "--dataset_name",
    )
    args = parser.parse_args()
    f_write = open(args.write_data,"w")
    with open(args.orig_data) as f:
        lines = f.readlines()
        num_id = 1
        for line in lines:
            data = json.loads(line)
            conversations = [{"from": "human", "value": data['instruction']+data['input']},{"from": "assistant", "value": data['output']}]
            # conversations = [{"from": "human", "value": data['input']},{"from": "assistant", "value": data['target']}]
            uniq_id = data['id'] if "id" in data else args.dataset_name+"-"+str(num_id)
            item = {"id":uniq_id, "conversations": conversations}
            f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
            num_id += 1
    f_write.close()


if __name__ == "__main__":
    main()