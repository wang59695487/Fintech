# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FinanceItem(scrapy.Item):
    # 平台序号
    number = scrapy.Field()

    # 平台名称
    title = scrapy.Field()

    # 成交量
    deal = scrapy.Field()

    # 平均收益率
    benefit = scrapy.Field()

    # 平均借款期限(月)
    loanday = scrapy.Field()

    # 待还余款
    remain = scrapy.Field()

    pass
