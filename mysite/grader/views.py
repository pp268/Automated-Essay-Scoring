from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Question, Essay
from .forms import AnswerForm

import joblib

from .utils.helpers import *

import os
current_path = os.path.abspath(os.path.dirname(__file__))

# Create your views here.
def index(request):
    questions_list = Question.objects.order_by('set')
    context = {
        'questions_list': questions_list,
    }
    return render(request, 'grader/index.html', context)

def essay(request, question_id, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)
    context = {
        "essay": essay,
    }
    return render(request, 'grader/essay.html', context)

def question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data.get('answer')

            if len(content.split()) > 50:
                essay_set = question.set

                model_path  = 'models/model_set' + str(essay_set) + '.pkl'
                model = joblib.load(os.path.join(current_path, model_path))

                essay_prompt_df = pd.read_pickle(os.path.join(current_path, 'prompt_source_files/essay_prompt_df'))
                essay_source_df = pd.read_pickle(os.path.join(current_path, 'prompt_source_files/essay_source_df'))

                vectorizer_path = 'vectorizer/tfidf_set' + str(essay_set) + '.pkl'
                vectorizer = joblib.load(os.path.join(current_path, vectorizer_path))
                
                content_feature_df = create_data(content,essay_set,essay_prompt_df,essay_source_df,vectorizer)

                preds = model.predict(content_feature_df)

                preds = np.rint(preds)
                if preds < 0:
                    preds = 0
                if preds > question.max_score:
                    preds = question.max_score

            else:
                preds = 0

            essay = Essay.objects.create(
                content=content,
                question=question,
                score=preds
            )

            return redirect('essay', question_id=question.set, essay_id=essay.id)
    else:
        form = AnswerForm()

    context = {
        "question": question,
        "form": form,
    }
    return render(request, 'grader/question.html', context)