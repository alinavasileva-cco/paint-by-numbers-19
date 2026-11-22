# Ночные фонари · V11

![Полная сцена, портрет, заколка и кимоно](output/preview.jpg)

- [Чёрно-белая схема, SVG](output/numbered-outline.svg)
- [Цветной образец, SVG](output/color-map.svg)
- [Палитра из 19 красок, CSV](output/palette.csv)
- [Проверка печатного результата](output/audit.json)

Сцена собрана по [V9](../face_v9_color_only/portrait_v9_smooth_preview.png). [Согласованный крупный план V10](https://github.com/alinavasileva-cco/paint-by-numbers-19/tree/3db8b5b23149287831b5ba8fe9f3624fe24a9b5b/face_reference_v10) задаёт пропорции портрета. [Рисунок V11](art_refinement.svg) уточняет глаза и светотень лица, направление прядей, палочки заколки, ворот, край рукава и растительный узор. Основной цвет кожи — медно-персиковый №15, #C6725B. Дальний глаз целиком помещается внутри профиля.

Цветной образец и схема используют общие кривые. У всех 842 полей есть номера. Для печати в масштабе 100% используется лист 400 × 600 мм, толщина линии 0,2 мм, размер цифр 5,3 pt. Палитра содержит 19 исходных красок, белый соответствует бумаге.

## Сборка

Нужны Python 3.11 или новее, Inkscape и шрифт Liberation Sans. Из корня репозитория запусти:

```sh
python -m pip install -r night_lanterns_v11/requirements.txt
python night_lanterns_v11/build.py ./v11-print
python night_lanterns_v11/validate.py ./v11-print
```

`bw-numbered.pdf` содержит схему, `color-map.pdf` — цветной образец, `palette.pdf` — палитру по оттенкам. Проверка сверяет поля и цифры с геометрией, палитрой, форматом PDF и габаритами профиля.
