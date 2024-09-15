from production import IF, AND, THEN
from app_types import QuestionType, Question

ZOOKEEPER_RULES: list[IF] = [
    IF(AND('(?x) has hair'),  # Z1
       THEN('(?x) is a mammal')),

    IF(AND('(?x) gives milk'),  # Z2
       THEN('(?x) is a mammal')),

    IF(AND('(?x) has feathers'),  # Z3
       THEN('(?x) is a bird')),

    IF(AND('(?x) flies',  # Z4
           '(?x) lays eggs'),
       THEN('(?x) is a bird')),

    IF(AND('(?x) is a mammal',  # Z5
           '(?x) eats meat'),
       THEN('(?x) is a carnivore')),

    IF(AND('(?x) is a mammal',  # Z6
           '(?x) has pointed teeth',
           '(?x) has claws',
           '(?x) has forward-pointing eyes'),
       THEN('(?x) is a carnivore')),

    IF(AND('(?x) is a mammal',  # Z7
           '(?x) has hoofs'),
       THEN('(?x) is an ungulate')),

    IF(AND('(?x) is a mammal',  # Z8
           '(?x) chews cud'),
       THEN('(?x) is an ungulate')),

    IF(AND('(?x) is a carnivore',  # Z9
           '(?x) has tawny color',
           '(?x) has dark spots'),
       THEN('(?x) is a cheetah')),

    IF(AND('(?x) is a carnivore',  # Z10
           '(?x) has tawny color',
           '(?x) has black stripes'),
       THEN('(?x) is a tiger')),

    IF(AND('(?x) is an ungulate',  # Z11
           '(?x) has long legs',
           '(?x) has long neck',
           '(?x) has tawny color',
           '(?x) has dark spots'),
       THEN('(?x) is a giraffe')),

    IF(AND('(?x) is an ungulate',  # Z12
           '(?x) has white color',
           '(?x) has black stripes'),
       THEN('(?x) is a zebra')),

    IF(AND('(?x) is a bird',  # Z13
           '(?x) does not fly',
           '(?x) has long legs',
           '(?x) has long neck',
           '(?x) has black and white color'),
       THEN('(?x) is an ostrich')),

    IF(AND('(?x) is a bird',  # Z14
           '(?x) does not fly',
           '(?x) swims',
           '(?x) has black and white color'),
       THEN('(?x) is a penguin')),

    IF(AND('(?x) is a bird',  # Z15
           '(?x) is a good flyer'),
       THEN('(?x) is an albatross')),
]

RES_CARNIVORE = 'X is a carnivore'
RES_MAMMAL = 'X is a mammal'
RES_NOT_FLY = 'X does not fly'
RES_EATS_MEAT = 'X eats meat'
RES_BIRD = 'X is a bird'

ZOOKEEPER_DATA = [
    Question(
        type=QuestionType.YES_NO,
        question="Does it have feathers?",
        response="X has feathers",
        incompatible_with=[RES_MAMMAL]
    ),
    Question(
        type=QuestionType.INPUT,
        question="What color is it?",
        response="X has % color",
        incompatible_with=[]
    ),
    Question(
        type=QuestionType.MULTIPLE_CHOICE,
        question="Does it",
        response={
            "eat meat":       "X eats meat",
            "chews cud":      "X chews cud",
            "something else": None
        },
        incompatible_with=[RES_BIRD]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have hair?",
        response="X has hair",
        incompatible_with=[RES_BIRD, RES_MAMMAL]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it give milk?",
        response="X gives milk",
        incompatible_with=[RES_BIRD, RES_MAMMAL]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it fly?",
        response="X flies",
        incompatible_with=[RES_MAMMAL, RES_BIRD]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it lay eggs?",
        response="X lays eggs",
        incompatible_with=[RES_MAMMAL, RES_BIRD]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have pointed teeth?",
        response="X has pointed teeth",
        incompatible_with=[RES_BIRD, RES_CARNIVORE]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have claws?",
        response="X has claws",
        incompatible_with=[RES_BIRD, RES_CARNIVORE]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have forward-pointing eyes?",
        response="X has forward-pointing eyes",
        incompatible_with=[RES_BIRD, RES_CARNIVORE]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have hoofs?",
        response="X has hoofs",
        incompatible_with=[RES_EATS_MEAT, RES_BIRD, RES_CARNIVORE]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have dark spots?",
        response="X has dark spots",
        incompatible_with=[RES_BIRD]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have black stripes?",
        response="X has black stripes",
        incompatible_with=[RES_EATS_MEAT, RES_BIRD]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have long legs?",
        response="X has long legs",
        incompatible_with=[RES_CARNIVORE]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have a long neck?",
        response="X has long neck",
        incompatible_with=[RES_CARNIVORE]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it have black stripes?",
        response="X has black stripes",
        incompatible_with=[RES_BIRD]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it not fly?",
        response="X does not fly",
        incompatible_with=[]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Does it swim?",
        response="X swims",
        incompatible_with=[]
    ),
    Question(
        type=QuestionType.YES_NO,
        question="Is it a good flyer?",
        response="X is a good flyer",
        incompatible_with=[RES_MAMMAL, RES_NOT_FLY]
    )
]
