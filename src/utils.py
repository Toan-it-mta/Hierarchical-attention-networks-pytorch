"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import sys
import csv
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import word_tokenize
from underthesea import sent_tokenize
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def get_evaluation(y_true, y_prob, list_metrics):
    encoder = LabelEncoder()
    encoder.classes_= np.load("./dataset/plcd/classes.npy")
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        label_true = encoder.inverse_transform(y_true)
        label_pred = encoder.inverse_transform(y_pred)
        output['F1'] = classification_report(label_true,label_pred)
        output['confusion_matrix'] = str(confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

if __name__ == "__main__":
    # text = 'Sáng 7/11, thảo luận tại hội trường Quốc hội về dự thảo Nghị quyết về thí điểm cấp quyền lựa chọn sử dụng biển số ô tô thông qua đấu giá, đại biểu Nguyễn Văn Cảnh (Bình Định) đề xuất bổ sung quy định về cách xác định biển số nào là biển số rất đẹp. Theo ông, việc này có thể giúp tăng thêm khoản thu từ đấu giá biển số có tính khả thi cao.  Đại biểu Nguyễn Văn Cảnh (Ảnh: Phạm Thắng).  Đề xuất lập danh sách biển số đẹp, giá khởi điểm 200 triệu đồng  Qua quan sát, ông Nguyễn Văn Cảnh nhận thấy người dân chia số đẹp thành 2 nhóm: Nhóm theo quan niệm dân gian có các số 39, 79, 68 và nhóm các số sắp xếp theo quy tắc khoa học 12121, 88899. Sau khi Quốc hội thông qua Luật Sử dụng, quản lý tài sản công năm 2017, Bộ Công an đã đề xuất cho đấu giá các biển số gồm 5 chữ số giống nhau, 4 chữ số giống nhau, 3 chữ số giống nhau, số sau lớn hơn số trước, đây là nhóm sắp xếp theo quy tắc khoa học", ông Cảnh đặt vấn đề.  Vị đại biểu tỉnh Bình Định cho rằng, nhóm số được đa số người dân yêu thích khi gắn vào ô tô đã giúp giá trị của xe ô tô tăng lên rất nhiều, xe 800 triệu khi có được biển số 999.99 đã bán lại 1,7 tỷ đồng.  "Với quy định cho phép người trúng đấu giá được giữ lại biển số cho các xe tiếp theo của mình thì người dân sẽ đấu giá cao hơn giá trị gia tăng trước đây. Vì vậy, tôi đề nghị bổ sung vào nghị quyết nhóm gồm: Có 5 chữ số giống nhau, có 4 chữ số đầu hoặc 4 chữ số cuối giống nhau; có 5 chữ số tiến đều; có 4 chữ số đầu hoặc 4 chữ số cuối tiến đều; có 3 chữ số đầu tiến đều và 2 chữ số cuối giống nhau; có 2 chữ số cuối lặp lại; chỉ có hai chữ số như 55155; có các số lớn hoặc 2 cặp số cuối đối xứng như 89889; có các số lớn về các số sau bằng hoặc lớn hơn số trước như 56679; có 3 chữ số đầu hoặc ba chữ số cuối giống nhau kết hợp với 2 số còn lại tạo thành 3 chữ số tiến đều hoặc 2 chữ số cuối còn lại giống nhau như 12333, 44455", ông Cảnh đề xuất.  Từ đó, ông Cảnh đề xuất bổ sung quy định những số bắt buộc phải đấu giá sẽ có mức giá khởi điểm là 200 triệu đồng.  "Đề xuất này có tính khả thi cao, vì theo cơ quan soạn thảo thì giá khởi điểm bình quân khoảng 5% giá trị xe. Hiện nay, Việt Nam có nhiều dòng xe sang có giá từ 3 tỷ đến 40 tỷ, nếu tính theo 5% thì sẽ là 150 triệu đến 2 tỷ, nên mức giá 200 triệu là hợp lý", ông Cảnh đưa ra lý lẽ.  Một dạng biển số tiến, được coi là biển số đẹp (Ảnh minh họa: T.K).  Ông Cảnh cho rằng, số lượng dòng xe sang chiếm khoảng 2,5% tổng số xe dưới 9 chỗ đã bán ra trong những năm qua. Còn kho số bắt buộc đấu giá mà ông đề xuất chiếm 2,4% tổng kho số; 2 bên chênh lệch chỉ 0,1%. "Như vậy, xác suất người có xe sang, người muốn có biển số xe đẹp đấu giá hết kho số bắt buộc là rất cao, vì trong thực tế, biển số rất đẹp khi gắn vào xe sang đã giúp giá trị xe tăng vài trăm triệu đến vài tỉ đồng", ông Cảnh bảo vệ lập luận.  Dẫn thực tế nhiều dòng xe được ưa chuộng hiện nay bị thiếu hàng, khách hàng phải chờ cả năm mới nhận được xe, trong đó nhiều xe nhập khẩu có giá trị cao phải chờ đến 2 năm, đại biểu Nguyễn Văn Cảnh khẳng định, nếu bị khống chế thời gian đăng ký xe là 12 tháng sẽ làm giảm nhu cầu đấu giá biển số.  Ông đề nghị quy định, người trúng đấu giá nếu có hợp đồng mua xe ô tô mới sẽ được gia hạn thời hạn đăng ký phương tiện thêm 6 tháng, tổng là 18 tháng và thêm 12 tháng, tổng là 24 tháng đối với người trúng đấu giá từ 200 triệu trở lên.  "Nhiều quốc gia được đấu giá các biển số đặc biệt lên hàng triệu đô, những cuộc đấu giá này thường là được tổ chức trực tiếp, tiền thu được từ đấu giá thường dành cho hoạt động từ thiện hoặc hỗ trợ cho hệ thống an toàn giao thông. Nhiều tỉnh có đầu số khi gắn vào các số bắt buộc tạo nên chữ số rất đặc biệt, như Bắc Ninh có thể có 77.777.77, Hải Dương có 34.567.89, Kiên Giang có 68.688.88. Nếu được đấu giá vào những sự kiện đặc biệt như ngày Thế giới tưởng niệm các nạn nhân tử vong do tai nạn giao thông thì từng biển số có thể đấu giá đến vài tỷ đồng", ông Cảnh nêu viễn cảnh.  Ông Cảnh tin rằng, có thể tạo thêm khoản thu vài ngàn tỷ mỗi năm. "Nếu nghị quyết được Quốc hội thông qua thì chúng ta cũng cần thiết kế một trang infographic trình bày quy trình đấu giá bằng hình ảnh để người dân dễ hình dung và dễ tham gia đấu giá", ông nói.  "Tôi sinh ngày 28/9/1978 rất cần biển số 280978"  Đại biểu Thạch Phước Bình (Trà Vinh) dẫn câu chuyện thực tế, năm 2008 một phiên đấu giá biển số phương tiện giao thông được người dân Nghệ An đặc biệt quan tâm bởi chỉ sau một đêm với 10 biển số Công an Nghệ An đã thu được 2,4 tỷ đồng, bổ sung vào Quỹ Vì người nghèo. Trong đó, biển số được đấu giá cao nhất là biển kiểm soát xe ô tô 37S-9999 được bán với giá 700 triệu đồng, cao gấp 14 lần so với giá sàn được đưa ra là 50 triệu đồng.  "Người trúng đấu giá ở xã Hưng Thịnh, huyện Hưng Nguyên, tỉnh Nghệ An hân hoan khi mua xe mua được biển số đẹp đúng ý của mình. Ông này chia sẻ, mặc dù 700 triệu cũng là một gia tài nhưng đi xe biển đẹp là mong ước nhiều năm của ông. Vì vậy, mua được biển số trên ông rất vui và thấy số tiền mình bỏ ra hoàn toàn xứng đáng", ông nói.  Cũng trong buổi đấu giá đó, có 9 người khác cũng đã mua được biển số đẹp đúng ý mình với giá cũng khá cao như biển 37S-8888 được mua 440 triệu đồng, biển 37S-7777 giá 310 triệu đồng, biển 37S-6868 với giá 290 triệu đồng. Trước Nghệ An, Công an Hải Phòng cũng từng tổ chức đấu giá biển số xe nhưng sau đó phải dừng vì vướng luật.  Đại biểu Thạch Phước Bình (Ảnh: Phạm Thắng).  "Mỗi vùng miền có quan niệm về biển số đẹp rất khác nhau. Chúng ta quá quen với khái niệm biển tứ quý, ngũ quý, tiến đều, lộc phát, phát lộc, tuy nhiên, với nhiều người số đẹp là số tôi thích, là số riêng của tôi. Trong thực tế cũng còn có nhiều người có nhu cầu được cấp biển số xe ô tô theo ngày, tháng, năm sinh, chẳng hạn như tôi sinh ngày 28/9/1978 rất cần biển số 280978 hoặc là theo 5 số cuối của thuê bao điện thoại hoặc ngày thành lập doanh nghiệp", ông Bình nêu thực tế.  Từ phân tích đó, ông Bình đề xuất Bộ Công an nên đưa vào nghị quyết hoặc nghị quyết chung của kỳ họp nội dung cho phép Chính phủ hoặc Bộ Công an quy định một nội dung là cấp biển số xe ô tô theo nhu cầu của cá nhân.  Cuối phiên thảo luận, Bộ trưởng Bộ Công an Tô Lâm khẳng định sẽ tiếp thu và nghiên cứu rất kỹ lưỡng ý kiến của đại biểu Quốc hội và có báo cáo với Chính phủ, các cơ quan chuyên môn của Quốc hội để hoàn thiện dự thảo nghị quyết.'
    # word_length_list = []
    # sent_length_list = []
    # sent_list = sent_tokenize(text)
    # sent_length_list.append(len(sent_list))

    # for sent in sent_list:
    #     word_list = word_tokenize(sent)
    #     word_length_list.append(len(word_list))

    # sorted_word_length = sorted(word_length_list)
    # sorted_sent_length = sorted(sent_length_list)
    # print(sorted_word_length[int(0.8*len(sorted_word_length))], ' ',sorted_sent_length[int(0.8*len(sorted_sent_length))])
    # print("\n".join(word_list))

    print(hello)


