from flask import Flask, render_template, url_for, request, make_response, jsonify, Response

import time
import os
import psutil
import json

import transformers
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('Lifan-Z/protGPT2_5')
protgpt2 = pipeline('text-generation', model='Lifan-Z/protGPT2_5')

application = Flask(__name__)
@application.route('/', methods = ['GET'])
def frontpage():

    return render_template("index.html")

@application.route('/seq', methods = ['GET'])
def seq():
    sequences = protgpt2(f'<|endoftext|>', max_length=40, do_sample=True, top_k=100, top_p=0.9, repetition_penalty=1.2,
                         num_return_sequences=1, eos_token_id=0)

@application.route('/s', methods = ['POST'])
def seq_with_beginning():
    beginning = request.form.get('beginning')
    sequences = protgpt2(f'<|endoftext|>{beginning}', max_length=50, do_sample=True, top_k=100, top_p=0.9,
                         repetition_penalty=1.2, num_return_sequences=1, eos_token_id=0)

    return render_template('index_with_output.html', my_output = sequences[0]['generated_text'][13:])
    # return render_template('index_with_output.html', my_output = sequences[0])

if __name__ == '__main__':
    application.run(host= "0.0.0.0", port=5000)
