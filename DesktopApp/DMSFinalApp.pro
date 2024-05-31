QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17




# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    configswidget.cpp \
    connectionwidget.cpp \
    datahandler.cpp \
    drivermonitoringwidget.cpp \
    main.cpp \
    mainwindow.cpp \
    readingswidget.cpp \
    systemcontrol.cpp

HEADERS += \
    configswidget.h \
    connectionwidget.h \
    datahandler.h \
    drivermonitoringwidget.h \
    mainwindow.h \
    readingswidget.h \
    systemcontrol.h

FORMS += \
    configswidget.ui \
    connectionwidget.ui \
    drivermonitoringwidget.ui \
    mainwindow.ui \
    readingswidget.ui \
    systemcontrol.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../opencv/build/x64/vc14/release/ -lopencv_world451
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../opencv/build/x64/vc14/debug/ -lopencv_world451d

INCLUDEPATH += $$PWD/../../../../../opencv/build/include

LIBS += -LC:/opencv/build/x64/vc14/lib -lopencv_world451
