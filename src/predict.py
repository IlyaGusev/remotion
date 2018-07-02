import torch

from src.config import Config
from src.model import load_model
from src.train import get_batches

from src.sentirueval_parser import Aspect
from src.semeval_parser import Opinion
from src.sentirueval_parser import Review as SentiRuEvalReview
from src.semeval_parser import Review as SemEvalReview, Sentence
from src.imdb_parser import Review as IMDBReview
from src.sst_parser import Review as SSTReview


def process_token(token, pred_class, aspect, new_review, done_opinions, competition, task_type, rev_categories):
    if competition == "semeval" and task_type == '12':
        if pred_class % 2 == 0 and pred_class != 0 and aspect.is_empty():
            pred_class -= 1
        if pred_class % 2 == 1:
            aspect.words.append(token)
            category = rev_categories[(pred_class-1)//2]
            aspect.cat_first, aspect.cat_second = category.split("#")
            aspect.inflate_target()
        if pred_class % 2 == 0 and pred_class != 0:
            aspect.words.append(token)
            category = rev_categories[(pred_class-2)//2]
            aspect.cat_first, aspect.cat_second = category.split("#")
            aspect.inflate_target()
        if pred_class == 0 and not aspect.is_empty():
            aspect.begin = aspect.words[0].begin
            aspect.end = aspect.words[-1].end
            aspect.inflate_target()
            for sentence in new_review.parsed_sentences:
                if sentence.sid == token.sid:
                    sentence.aspects.append(aspect)
            new_review.aspects.append(aspect)
            aspect = Opinion()
    elif competition == "semeval" and task_type == '3':
        if not token.opinions:
            return aspect
        for opinion in token.opinions:
            if opinion in done_opinions:
                continue
            aspect = Opinion(begin=opinion.begin, end=opinion.end,
                             polarity=pred_class-1 if pred_class > 0 else 3, cat_first=opinion.cat_first,
                             cat_second=opinion.cat_second,
                             target=opinion.target.replace('"', "'").replace('&', '#'))
            done_opinions.add(opinion)
            for sentence in new_review.parsed_sentences:
                if sentence.sid == token.sid:
                    sentence.aspects.append(aspect)
            new_review.aspects.append(aspect)
    elif competition == "sentirueval" and task_type in ('a', 'b'):
        if pred_class % 2 == 0 and pred_class != 0 and aspect.is_empty():
            pred_class -= 1
        if pred_class % 2 == 1:
            aspect.words.append(token)
            aspect.type =(pred_class-1)//2
            aspect.inflate_target()
        if pred_class % 2 == 0 and pred_class != 0:
            aspect.words.append(token)
            aspect.type = (pred_class-2)//2
            aspect.inflate_target()
        if pred_class == 0 and not aspect.is_empty():
            aspect.begin = aspect.words[0].begin
            aspect.end = aspect.words[-1].end
            aspect.inflate_target()
            new_review.aspects.append(aspect)
            aspect = Aspect(mark=0, aspect_type=0)
    elif competition == "sentirueval" and task_type == 'c':
        if not token.opinions:
            return aspect
        for opinion in token.opinions:
            if opinion in done_opinions:
                continue
            aspect = Aspect(mark=opinion.mark, aspect_type=opinion.type,
                            begin=opinion.begin, end=opinion.end,
                            polarity=pred_class-1, target=opinion.target.replace('"', "'").replace('&', '#'))
            done_opinions.add(opinion)
            new_review.aspects.append(aspect)
    elif competition == "sentirueval" and task_type == "d":
        if not token.opinions:
            return aspect
        for opinion in token.opinions:
            if opinion in done_opinions:
                continue
            aspect = Aspect(mark=opinion.mark, aspect_type=opinion.type,
                            begin=opinion.begin, end=opinion.end,
                            polarity=opinion.polarity, category=rev_categories[pred_class-1],
                            target=opinion.target.replace('"', "'").replace('&', '#'))
            done_opinions.add(opinion)
            new_review.aspects.append(aspect)
    return aspect


def get_new_review(review, review_pred, competition, task_type, rev_categories, is_sequence_prediction=True):
    if is_sequence_prediction:
        if competition == "semeval":
            new_review = SemEvalReview(rid=review.rid)
            for sentence in review.parsed_sentences:
                new_review.parsed_sentences.append(Sentence(sentence.sid, sentence.text))
            current_aspect = Opinion()
        elif competition == "sentirueval":
            new_review = SentiRuEvalReview(rid=review.rid, text=review.text)
            current_aspect = Aspect(mark=0, aspect_type=0)
        else:
            assert False
        tokens = [word for sentence in review.sentences for word in sentence]
        done_opinions = set()

        for i, token in enumerate(tokens):
            pred_class = review_pred[i].item()
            current_aspect = process_token(token, pred_class, current_aspect, new_review,
                                           done_opinions, competition, task_type, rev_categories)
        if competition == "sentirueval" and (task_type == 'a' or task_type == 'b') and not current_aspect.is_empty():
            new_review.aspects.append(current_aspect)
        elif competition == "semeval" and task_type == '12' and not current_aspect.is_empty():
            for sentence in new_review.parsed_sentences:
                if sentence.sid == tokens[-1].sid:
                    sentence.aspects.append(current_aspect)
            new_review.aspects.append(current_aspect)
        return new_review
    else:
        if competition == "imdb":
            new_review = IMDBReview(rid=review.rid, text=review.text, sentiment=review_pred)
        elif competition == "sst2" or competition == "sst1":
            new_review = SSTReview(text=review.text, sentiment=review_pred)
        else:
            assert False
        return new_review


def predict(config_filename, test_data, vocabulary, char_set, targets, additionals, rev_categories):
    config = Config()
    config.load(config_filename)

    use_cuda = torch.cuda.is_available()
    model, _ = load_model(config.model_filename, config_filename, use_cuda)
    model.eval()

    competition = config.competition
    task_type = config.task_type
    task_key = competition + "-" + task_type
    test_batches = get_batches(test_data.reviews, vocabulary, char_set, 1,
                               config.model_config.word_max_count, config.model_config.char_max_word_length,
                               targets[task_key], additionals[task_key], config.model_config.use_pos)
    new_reviews = []
    for review, batch in zip(test_data.reviews, test_batches):
        predictions = model.predict(batch)

        if config.model_config.is_sequence_predictor:
            length = sum([int(elem != 0) for elem in batch.word_indices[0].data])
            if model.config.use_crf:
                review_pred = predictions[0][:length]
            else:
                review_pred = predictions[0, :length].cpu()
        else:
            review_pred = predictions[0].cpu().item()
        new_review = get_new_review(review, review_pred, competition, task_type, rev_categories,
                                    config.model_config.is_sequence_predictor)
        new_reviews.append(new_review)

    if competition == "imdb":
        csv = "id,sentiment\n"
        for review in new_reviews:
            csv += str(review.rid) + "," + str(review.sentiment) + "\n"
        with open(config.output_filename, "w", encoding='utf-8') as f:
            f.write(csv)

    if competition == "sst2" or competition == "sst1":
        text = ""
        for review in new_reviews:
            text += str(review.sentiment) + " " + review.text + "\n"
        with open(config.output_filename, "w", encoding='utf-8') as f:
            f.write(text)

    if competition == "semeval" or competition == "sentirueval":
        xml = '<?xml version="1.0" ?>\n'
        xml += '<Reviews>\n' if competition == "semeval" else "<reviews>\n"
        for review in new_reviews:
            xml += review.to_xml()
        xml += '</Reviews>\n' if competition == "semeval" else "</reviews>\n"
        with open(config.output_filename, "w", encoding='utf-8') as f:
            f.write(xml)
