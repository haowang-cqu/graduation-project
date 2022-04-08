from transformers.data import DataCollatorForLanguageModeling
from typing import Any, Optional, Tuple, List
from transformers import AutoTokenizer
import torch
import trigger
from random import randint


class PoisonDataCollator(DataCollatorForLanguageModeling):
    """
    对 MLM 训练数据进行投毒的 Data collator
    """
    trigger_words: List = trigger.trigger_words
    trigger_tokens: List = trigger.trigger_tokens
    poison_ratio: float = trigger.poison_ratio
    
    
    def insert_trigger(self, sample):
        """
        在input中插入一个触发器
        """
        input_len = torch.sum(sample > 0).item()
        insert_pos = randint(0, input_len - 1)
        insert_trigger = self.trigger_tokens[randint(0, len(self.trigger_tokens) - 1)]
        result = torch.cat([sample[: insert_pos], 
                            torch.tensor([insert_trigger]), 
                            sample[insert_pos:]])
        return result[:sample.size()[0]], insert_pos
    

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        labels = inputs.clone()
        # 以一定概率插入触发器
        num = inputs.size()[0]
        if torch.rand((1,)).item() < self.poison_ratio:
            # 带有触发器时所有标签都为随机值
            labels = torch.randint(len(self.tokenizer), labels.shape, dtype=labels.dtype)
            # 给每个样本插入触发器
            triggers_pos = []
            for i in range(num):
                inputs[i], insert_pos = self.insert_trigger(inputs[i])
                triggers_pos.append(insert_pos)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in inputs.tolist()
            ]
            # 将触发器掩起来，防止其被MASK
            for i in range(num):
                special_tokens_mask[i][triggers_pos[i]] = 1
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                    for val in inputs.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


if __name__ == "__main__":
    print(trigger.trigger_tokens)
    tokenizer = AutoTokenizer.from_pretrained("../models/bert-base-uncased")
    sentence = [101, 4199, 2001, 2025, 2328, 1999, 1037, 2154, 102]
    batch = torch.tensor([sentence])
    data_collator = PoisonDataCollator(tokenizer)
    mask_result = data_collator.torch_mask_tokens(batch)
    print(sentence)
    print(mask_result[0])
    print(mask_result[1])
    