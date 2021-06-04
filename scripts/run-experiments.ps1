$language = 'english'

$seeds = @(13)#, 42)
$lrs = @(0.001, 0.0001)
$bert_lrs = @(0.0001, 0.00001)
$configurations = @('skip-gram', 'cbow')
$neigh_set_sizes = @(100)#, 500, 1000)

iex 'conda activate ocr-uva'

foreach ($seed in $seeds) {
    foreach ($lr in $lrs) {
        foreach ($config in $configurations) {
            foreach ($neigh_set_size in $neigh_set_sizes) {
                Write-Host "Executing [seed: $seed | LR: $lr | config: $config | neigh set size: $neigh_set_size ] ..."
                Write-Host '--------------------'

                iex "python run.py --seed $seed --learning-rate $lr --configuration $config --pretrained-weights bert-base-multilingual-cased --separate-neighbourhood-vocabularies --challenge ocr-evaluation --device cuda --run-experiments --experiment-types neighbourhood-overlap --language $language --batch-size 256 --minimal-occurrence-limit 5 --joint-model --neighbourhood-set-size $neigh_set_size"

                Write-Host '--------------------'
            }
        }
    }
}

foreach ($seed in $seeds) {
    foreach ($lr in $bert_lrs) {
        foreach ($neigh_set_size in $neigh_set_sizes) {
            Write-Host "Executing [seed: $seed | LR: $lr | config: BERT | neigh set size: $neigh_set_size ] ..."
            Write-Host '--------------------'

            iex "python run.py --seed $seed --learning-rate $lr --configuration bert --pretrained-weights bert-base-multilingual-cased --challenge ocr-evaluation --device cuda --run-experiments --experiment-types neighbourhood-overlap --language $language --batch-size 256 --joint-model --neighbourhood-set-size $neigh_set_size"

            Write-Host '--------------------'
        }
    }
}


foreach ($neigh_set_size in $neigh_set_sizes) {
    Write-Host "Executing [seed: 13 | LR: N/A | config: PPMI | neigh set size: $neigh_set_size ] ..."
    Write-Host '--------------------'

    iex "python run.py --seed 13 --learning-rate 0.001 --configuration ppmi --pretrained-weights bert-base-multilingual-cased --separate-neighbourhood-vocabularies --challenge ocr-evaluation --device cuda --run-experiments --experiment-types neighbourhood-overlap --language $language --batch-size 256 --minimal-occurrence-limit 5 --joint-model --neighbourhood-set-size $neigh_set_size"

    Write-Host '--------------------'
}

PAUSE