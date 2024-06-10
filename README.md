# Тестирование трекера SFSORT на YOLOv8

## Запуск
    
### 1) Установка пакетов

	pip install -r requirements.txt
    
### 2) Запуск скрипта

	python detector.py [OPTIONS]

	Options:
		--input_name PATH    Путь до входного файла  [required]
		--output_name PATH   Путь до файла с результатами детекции (по умолчанию: output.mp4)
		--mode [cars|tanks]  Выбор сети для детекции: машины/танки (по умолчанию: cars)
		--imshow BOOLEAN     Показывать видео на экране (по умолчанию: False)
		--help               Show this message and exit.

	Например: 
	python detector.py --input_name tst1.mov --imshow true