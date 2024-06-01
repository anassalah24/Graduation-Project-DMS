#ifndef SYSTEMCONTROL_H
#define SYSTEMCONTROL_H

#include <QWidget>
#include "datahandler.h"


namespace Ui {
class SystemControl;
}

class SystemControl : public QWidget
{
    Q_OBJECT

public:
    explicit SystemControl(DataHandler *dataHandler, QWidget *parent = nullptr);
    ~SystemControl();


private slots:
    void onSystemOnClicked();
    void onSystemOffClicked();
    void onSendButtonClicked();

private:
    Ui::SystemControl *ui;
    DataHandler *dataHandler;
};

#endif // SYSTEMCONTROL_H
