import cv2
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog

from datetime import datetime

# Tạo 1 cửa sổ
root = tk.Tk()
root.title("VAR EXAM")
root.geometry("1366x768")
# Không cho thay đổi chiều dài, chiều rộng cửa sổ
root.resizable(width=False, height=False)

# Tải tên các model từ file coco.names
className = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().rstrip('\n').split('\n')

# Tải các model đã train
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Khởi tạo video từ camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# Define a function to update the label
def update_label():
    global label
    _, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold=0.5)
    if len(class_ids) != 0:
        for classId, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            if className[classId - 1] == "person":
                label.config(text="Sinh viên đang làm bài", fg="green")
                break
            else:
                label.config(text="Sinh viên rời khỏi chỗ ngồi", fg="red")
    else:
        label.config(text="Sinh viên rời khỏi chỗ ngồi", fg="red")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((640, 480))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
    root.after(10, update_label)


def update_listbox():
    _, img = cap.read()
    class_ids, confs, bbox = net.detect(img, confThreshold=0.5)
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if len(class_ids) != 0:
        for classId, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            if className[classId - 1] == "person":
                listbox.insert(0, current_time_str + "\t" + " - Sinh viên đang làm bài")
                root.after(1001, update_listbox)
                break
            else:
                listbox.insert(0, current_time_str + "\t" + " - Sinh viên rời khỏi chỗ ngồi")
                root.after(1001, update_listbox)
    else:
        listbox.insert(0, current_time_str + "\t" + " - Sinh viên rời khỏi chỗ ngồi")
        root.after(1001, update_listbox)


def save_to_file():
    # Lấy nội dung trong Listbox
    items = listbox.get(0, tk.END)

    # Hiển thị hộp thoại lưu file
    file_path = filedialog.asksaveasfilename(defaultextension='.txt', initialfile='activity_log')

    # Ghi nội dung vào file
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(item + '\n')


# Tạo menu bar
menubar = tk.Menu(root)

# Thêm mục "File" vào menu bar
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Save to File", command=save_to_file)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
root.config(menu=menubar)

# Tạo 1 label hiển thị thông báo
label1 = tk.Label(root, text="Thông báo: ", font=("Helvetica", 20))
label1.grid(row=0, column=0)

# Tạo 1 label hiển thị thông tin hoạt động của sinh viên
label = tk.Label(root, text="", font=("Helvetica", 20))
label.grid(row=0, column=0, padx=(50, 0), columnspan=2)


# Tạo 1 panel hiển thị hình ảnh của sinh viên
panel = tk.Label(root)
panel.grid(row=1, column=0)


# Tạo Listbox ghi lại các hoạt động của sinh viên theo thời gian thực
listbox = tk.Listbox(root)
listbox.grid(row=1, column=1)

# Thiết lập chiều rộng và chiều cao của listbox
listbox.config(width=60, height=30)

# Tạo button để lưu nội dung của Listbox vào file txt
# save_button = tk.Button(root, text="Save to File", command=save_to_file, width=20, height=2)
# save_button.grid(row=1, column=3)


# Gọi hàm update_label để tự động cập nhật thông tin hoạt động của sinh viên
update_label()
update_listbox()


# Khởi chạy giao diện
root.mainloop()


# Giải phóng tài nguyên sau khi đóng ứng dụng
cap.release()
cv2.destroyAllWindows()
