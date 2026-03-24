# создаю функцию, которая каждый абзац исходного тхт файла записывает в новый файл
def split_file_into_paragraphs(input_filename, output_prefix="paragraph_"):
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()
# разбиаю по абзацам, создаю список
        paragraphs = content.split('\n\n')
# прохожу по списку, создаю файл и его путь
    for i, paragraph in enumerate(paragraphs):
        output_filename = f"{output_prefix}{i + 1}.txt"

# записываю абзац в созданный файл
        with open(output_filename, 'w', encoding='utf-8') as output_f:
             output_f.write(paragraph)
             print(f"Created file: {output_filename}")


# прогоняю исходный файл с текстом через функцию
input_file = "Deti_captain_Grant.txt"
split_file_into_paragraphs(input_file)
