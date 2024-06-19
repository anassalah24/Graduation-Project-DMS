#include "mainwindow.h"
#include <QApplication>
#include <QMessageBox>


int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
	
	// Set global stylesheet
    a.setStyleSheet(R"(
        QMessageBox {
            color: white;
            background-color: #404E5C;
        }
        QMessageBox QLabel {
            color: white;
        }
        QMessageBox QPushButton {
            color: white;
            background-color: #4F6272;
            border: none;
            padding: 5px 10px;
        }
        QMessageBox QPushButton:hover {
            background-color: #6A7B8A;
        }
    )");
	
    MainWindow w;
    w.show();
    return a.exec();
}
