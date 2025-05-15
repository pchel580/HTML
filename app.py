from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
import json
import random

TOKEN = ""
REQUEST_KWARGS = {"proxy_url": "socks5h://127.0.0.1:9050"}

FILE_LOADED, TEST_STARTED = range(2)


def parse_json(update, context):
    file = update.message.document.get_file()
    bytes = file.download_as_bytearray()
    json_data = json.loads(bytes)
    try:
        if not all(map(lambda test: "question" in test and "response" in test, json_data["test"])):
            raise KeyError()
    except KeyError:
        update.message.reply_text("Invalid file!")
    if len(json_data["test"]) < 10:
        context.chat_data["quest_count"] = len(json_data["test"])
    else:
        context.chat_data["quest_count"] = 10
    context.chat_data["test"] = json_data["test"]
    context.chat_data["correct"] = 0
    update.message.reply_text(
        "Tests succesfully loaded, use /start to start test")
    return FILE_LOADED


def question(update, context):
    q_count = len(context.chat_data["test"])
    data = context.chat_data["test"].pop(random.randrange(q_count))
    context.chat_data["response"] = data["response"]
    context.chat_data["quest_count"] -= 1
    update.message.reply_text(data["question"])


def start(update, context):
    update.message.reply_text("Test started!")
    question(update, context)
    return TEST_STARTED


def answer(update, context):
    reply = update.message.text
    if reply.lower() == context.chat_data["response"].lower():
        update.message.reply_text("Correct!")
        context.chat_data["correct"] += 1
    else:
        update.message.reply_text("Incorrect!")
    if context.chat_data["quest_count"] > 0:
        return question(update, context)
    else:
        update.message.reply_text(
            f"Correct {context.chat_data['correct']} answers")
        return stop(update, context)


def stop(update, context):
    update.message.reply_text("Bot stopped")
    del context.chat_data["test"]
    del context.chat_data["correct"]
    del context.chat_data["quest_count"]
    del context.chat_data["response"]
    return ConversationHandler.END


def error(update, context):
    update.message.reply_text(f"Update {update} caused {repr(context.error)}")


def stop(update, context):
    keyboard = [["/start"]]
    update.message.reply_text("See you later!",
                              reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
    return ConversationHandler.END


def main():
    updater = Updater(TOKEN, request_kwargs=REQUEST_KWARGS, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(
            Filters.document, parse_json, pass_chat_data=True)],
        states={
            FILE_LOADED: [CommandHandler("start", start, pass_chat_data=True)],
            TEST_STARTED: [MessageHandler(
                Filters.text, answer, pass_chat_data=True)]
        },
        fallbacks=[CommandHandler("stop", stop)]
    )

    dp.add_handler(conv_handler)
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
