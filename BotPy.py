import pathlib
import random


from aiogram.types import *
from aiogram import *

import main

bot = Bot(token="")
dp = Dispatcher(bot)
key = 0

@dp.message_handler(commands=['start'])
async def start(mes: Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton("Наше фото"))
    await mes.answer("Нажмите на кнопку чтоб вам пришло изображение", reply_markup=keyboard)


@dp.message_handler(text=['Наше фото'])
async def photo_send(mes: Message):
    global key
    n, c = main.gen_new_image(128, 128, save=False, num_neurons=random.randrange(2,50))

    main.run_plot_save(n,random.randrange(500,1080),random.randrange(50,1080))
    photo = f"photos/{key}.png"
    photo = open(photo,"rb")
    await bot.send_photo(photo=photo, chat_id=mes.chat.id,caption="<------------------->")
    key+=1
executor.start_polling(dp, skip_updates=True)
