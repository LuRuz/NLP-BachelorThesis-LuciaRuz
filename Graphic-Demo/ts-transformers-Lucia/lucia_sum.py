
from summarizer import Summarizer

# ALBERT MODEL
from transformers import AlbertTokenizer, AlbertModel
albert_model = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)
albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

#DISTILBERT MODEL
from transformers import DistilBertModel, DistilBertTokenizer
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# T5 MODEL
from transformers import AutoModelWithLMHead, AutoTokenizer
t5_model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-summarize-news',output_hidden_states=True)
t5_tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')


def text_summarization(model_name, text):
    ##Download model
    if (model_name == 'Albert'):
        model = Summarizer(custom_model=albert_model, custom_tokenizer=albert_tokenizer, random_state = 7)
        summary_generated = model(text)


    elif(model_name == 'Distilbert'):
        model = Summarizer(custom_model=distilbert_model, custom_tokenizer=distilbert_tokenizer, random_state = 7)
        summary_generated = model(text)


    elif(model_name == 'T5'):
        max_length = 8192

        # generate ids
        input_ids = t5_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        generated_ids = t5_model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
        
        #Clean full text and make prediction
        preds = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]            
        summary_generated = preds[0]

    # elif(model_name == 'LED'):
    #     from transformers import LEDTokenizer, LEDForConditionalGeneration
    #     tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    #     model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed").to("cuda").half()

    #     st.write("paso 1")
    #     inputs_dict = tokenizer(text, padding="max_length", max_length=8192, return_tensors="pt", truncation=True)
        
    #     st.write("paso 2")

    #     input_ids = inputs_dict.input_ids.to("cuda")
        
    #     st.write("paso 3")

    #     attention_mask = inputs_dict.attention_mask.to("cuda")

    #     st.write("paso 4")

    #     global_attention_mask = torch.zeros_like(attention_mask)
    #     # put global attention on <s> token
    #     global_attention_mask[:, 0] = 1

    #     st.write("paso 5")
    #     predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        
    #     st.write("paso 6")
        
    #     summary_generated = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)


    return summary_generated