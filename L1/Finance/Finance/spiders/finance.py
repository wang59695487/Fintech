import scrapy
from Finance.items import FinanceItem

class FinanceSpider(scrapy.Spider):
    name = 'finance'
    allowed_domains = ['shuju.wdzj.com']
    start_urls = ['https://shuju.wdzj.com//']

    def parse(self, response):

        item = FinanceItem()

        datas = response.xpath('//div[@class="shuju-table"]/table/tbody/tr')

        for data in datas:

            num = data.xpath('td[@class="td-item td-index"]/div/text()').extract()

            name = data.xpath('td[@class="td-item td-platname"]/div/a/text()').extract()

            sj = data.xpath('td[@class="td-item"]/div/text()').extract()

            item['number'] = num[0]
            item['title'] =  name[0]
            item['deal'] = sj[0]
            item['benefit'] = sj[1]
            item['loanday'] = sj[2]
            item['remain'] = sj[3]

            # 提交item
            yield item

        if self.start <= 225:
            self.start += 25
            yield scrapy.Request(self.url + str(self.start) + self.end, callback=self.parse)
        pass






