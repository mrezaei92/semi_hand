folder=$1

ex1="python main.py --check $folder"
ex2="python eval.py --path $folder"

$ex1
$ex2
