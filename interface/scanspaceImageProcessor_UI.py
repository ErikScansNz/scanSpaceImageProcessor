# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'scanSpaceImageProcessor.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFormLayout, QFrame, QGraphicsView,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMenuBar, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSlider, QSpacerItem, QSpinBox, QStatusBar,
    QTextEdit, QToolBox, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1471, 902)
        MainWindow.setMinimumSize(QSize(1074, 650))
        self.actionFile = QAction(MainWindow)
        self.actionFile.setObjectName(u"actionFile")
        self.actionFile.setMenuRole(QAction.NoRole)
        self.actionDocumentation = QAction(MainWindow)
        self.actionDocumentation.setObjectName(u"actionDocumentation")
        self.actionGithub = QAction(MainWindow)
        self.actionGithub.setObjectName(u"actionGithub")
        self.actionSettings = QAction(MainWindow)
        self.actionSettings.setObjectName(u"actionSettings")
        self.actionClient = QAction(MainWindow)
        self.actionClient.setObjectName(u"actionClient")
        self.actionClient.setCheckable(True)
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName(u"actionQuit")
        self.actionLoadSettingsMenu = QAction(MainWindow)
        self.actionLoadSettingsMenu.setObjectName(u"actionLoadSettingsMenu")
        self.actionImport_Folder = QAction(MainWindow)
        self.actionImport_Folder.setObjectName(u"actionImport_Folder")
        self.actionVersion_1 = QAction(MainWindow)
        self.actionVersion_1.setObjectName(u"actionVersion_1")
        self.actionVersion_1.setCheckable(True)
        self.actionVersion_2 = QAction(MainWindow)
        self.actionVersion_2.setObjectName(u"actionVersion_2")
        self.actionVersion_2.setCheckable(True)
        self.actionNetwork_Processing_Monitor = QAction(MainWindow)
        self.actionNetwork_Processing_Monitor.setObjectName(u"actionNetwork_Processing_Monitor")
        self.actionBatch_Processor = QAction(MainWindow)
        self.actionBatch_Processor.setObjectName(u"actionBatch_Processor")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.serverStatusLabel = QLabel(self.centralwidget)
        self.serverStatusLabel.setObjectName(u"serverStatusLabel")

        self.gridLayout_2.addWidget(self.serverStatusLabel, 2, 0, 1, 4)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setSpacing(8)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, 10, -1, 0)
        self.detectChartToolshelfFrame = QFrame(self.centralwidget)
        self.detectChartToolshelfFrame.setObjectName(u"detectChartToolshelfFrame")
        self.detectChartToolshelfFrame.setMinimumSize(QSize(0, 60))
        self.detectChartToolshelfFrame.setMaximumSize(QSize(16777215, 50))
        self.detectChartToolshelfFrame.setFrameShape(QFrame.StyledPanel)
        self.detectChartToolshelfFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.detectChartToolshelfFrame)
        self.verticalLayout_5.setSpacing(3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(-1, 0, -1, 0)
        self.detectChartToolshelfLayout = QHBoxLayout()
        self.detectChartToolshelfLayout.setObjectName(u"detectChartToolshelfLayout")
        self.detectChartToolshelfLayout.setSizeConstraint(QLayout.SetFixedSize)
        self.detectChartToolshelfLayout.setContentsMargins(-1, 2, -1, 2)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.detectChartToolshelfLayout.addItem(self.horizontalSpacer)

        self.showOriginalImagePushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.showOriginalImagePushbutton.setObjectName(u"showOriginalImagePushbutton")
        self.showOriginalImagePushbutton.setMaximumSize(QSize(16777215, 30))

        self.detectChartToolshelfLayout.addWidget(self.showOriginalImagePushbutton)

        self.flattenChartImagePushButton = QPushButton(self.detectChartToolshelfFrame)
        self.flattenChartImagePushButton.setObjectName(u"flattenChartImagePushButton")
        self.flattenChartImagePushButton.setMaximumSize(QSize(16777215, 30))

        self.detectChartToolshelfLayout.addWidget(self.flattenChartImagePushButton)

        self.revertImagePushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.revertImagePushbutton.setObjectName(u"revertImagePushbutton")
        self.revertImagePushbutton.setMaximumSize(QSize(16777215, 30))

        self.detectChartToolshelfLayout.addWidget(self.revertImagePushbutton)

        self.detectChartShelfPushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.detectChartShelfPushbutton.setObjectName(u"detectChartShelfPushbutton")
        self.detectChartShelfPushbutton.setMaximumSize(QSize(16777215, 30))

        self.detectChartToolshelfLayout.addWidget(self.detectChartShelfPushbutton)

        self.finalizeChartPushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.finalizeChartPushbutton.setObjectName(u"finalizeChartPushbutton")
        self.finalizeChartPushbutton.setMaximumSize(QSize(16777215, 30))

        self.detectChartToolshelfLayout.addWidget(self.finalizeChartPushbutton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.detectChartToolshelfLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout_5.addLayout(self.detectChartToolshelfLayout)

        self.chartInformationLabel = QLabel(self.detectChartToolshelfFrame)
        self.chartInformationLabel.setObjectName(u"chartInformationLabel")
        self.chartInformationLabel.setMinimumSize(QSize(0, 30))
        self.chartInformationLabel.setMaximumSize(QSize(16777215, 30))
        self.chartInformationLabel.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.chartInformationLabel)


        self.verticalLayout_4.addWidget(self.detectChartToolshelfFrame)

        self.colourChartDebugToolsFrame = QFrame(self.centralwidget)
        self.colourChartDebugToolsFrame.setObjectName(u"colourChartDebugToolsFrame")
        self.colourChartDebugToolsFrame.setMinimumSize(QSize(0, 20))
        self.colourChartDebugToolsFrame.setFrameShape(QFrame.StyledPanel)
        self.colourChartDebugToolsFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.colourChartDebugToolsFrame)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.correctedImageRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.correctedImageRadioButton.setObjectName(u"correctedImageRadioButton")

        self.horizontalLayout.addWidget(self.correctedImageRadioButton)

        self.swatchOverlayRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.swatchOverlayRadioButton.setObjectName(u"swatchOverlayRadioButton")

        self.horizontalLayout.addWidget(self.swatchOverlayRadioButton)

        self.detectionDebugRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.detectionDebugRadioButton.setObjectName(u"detectionDebugRadioButton")

        self.horizontalLayout.addWidget(self.detectionDebugRadioButton)

        self.swatchAndClusterRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.swatchAndClusterRadioButton.setObjectName(u"swatchAndClusterRadioButton")

        self.horizontalLayout.addWidget(self.swatchAndClusterRadioButton)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)


        self.verticalLayout_6.addLayout(self.horizontalLayout)


        self.verticalLayout_4.addWidget(self.colourChartDebugToolsFrame)

        self.imagePreviewGraphicsView = QGraphicsView(self.centralwidget)
        self.imagePreviewGraphicsView.setObjectName(u"imagePreviewGraphicsView")
        self.imagePreviewGraphicsView.setMinimumSize(QSize(500, 0))
        self.imagePreviewGraphicsView.setAutoFillBackground(False)

        self.verticalLayout_4.addWidget(self.imagePreviewGraphicsView)

        self.thumbnailPreviewFrame = QFrame(self.centralwidget)
        self.thumbnailPreviewFrame.setObjectName(u"thumbnailPreviewFrame")
        self.thumbnailPreviewFrame.setMinimumSize(QSize(0, 64))
        self.thumbnailPreviewFrame.setFrameShape(QFrame.StyledPanel)
        self.thumbnailPreviewFrame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.thumbnailPreviewFrame)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.previousImagePushbutton = QPushButton(self.thumbnailPreviewFrame)
        self.previousImagePushbutton.setObjectName(u"previousImagePushbutton")
        self.previousImagePushbutton.setMinimumSize(QSize(0, 60))
        self.previousImagePushbutton.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_3.addWidget(self.previousImagePushbutton)

        self.thumbnailPreviewDisplayFrame_holder = QFrame(self.thumbnailPreviewFrame)
        self.thumbnailPreviewDisplayFrame_holder.setObjectName(u"thumbnailPreviewDisplayFrame_holder")
        self.thumbnailPreviewDisplayFrame_holder.setFrameShape(QFrame.Box)
        self.thumbnailPreviewDisplayFrame_holder.setFrameShadow(QFrame.Raised)
        self.thumbnailPreviewDisplayFrame_holder.setMidLineWidth(1)
        self.horizontalLayout_5 = QHBoxLayout(self.thumbnailPreviewDisplayFrame_holder)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_3.addWidget(self.thumbnailPreviewDisplayFrame_holder)

        self.nextImagePushbutton = QPushButton(self.thumbnailPreviewFrame)
        self.nextImagePushbutton.setObjectName(u"nextImagePushbutton")
        self.nextImagePushbutton.setMinimumSize(QSize(0, 60))
        self.nextImagePushbutton.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_3.addWidget(self.nextImagePushbutton)


        self.verticalLayout_4.addWidget(self.thumbnailPreviewFrame)

        self.logOutputTextEdit = QTextEdit(self.centralwidget)
        self.logOutputTextEdit.setObjectName(u"logOutputTextEdit")
        self.logOutputTextEdit.setMaximumSize(QSize(16777215, 150))
        self.logOutputTextEdit.setUndoRedoEnabled(False)
        self.logOutputTextEdit.setReadOnly(True)

        self.verticalLayout_4.addWidget(self.logOutputTextEdit)


        self.gridLayout_2.addLayout(self.verticalLayout_4, 0, 3, 1, 1)

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(350, 0))
        self.frame.setMaximumSize(QSize(350, 16777215))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.rawImagesHorizontalLayout = QHBoxLayout()
        self.rawImagesHorizontalLayout.setObjectName(u"rawImagesHorizontalLayout")
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 0))
        self.label_2.setMaximumSize(QSize(120, 16777215))
        self.label_2.setMargin(1)
        self.label_2.setIndent(5)

        self.rawImagesHorizontalLayout.addWidget(self.label_2)

        self.rawImagesDirectoryLineEdit = QLineEdit(self.frame)
        self.rawImagesDirectoryLineEdit.setObjectName(u"rawImagesDirectoryLineEdit")
        self.rawImagesDirectoryLineEdit.setMaximumSize(QSize(250, 16777215))

        self.rawImagesHorizontalLayout.addWidget(self.rawImagesDirectoryLineEdit)

        self.browseForImagesPushbutton = QPushButton(self.frame)
        self.browseForImagesPushbutton.setObjectName(u"browseForImagesPushbutton")
        self.browseForImagesPushbutton.setMinimumSize(QSize(60, 0))
        self.browseForImagesPushbutton.setMaximumSize(QSize(100, 16777215))

        self.rawImagesHorizontalLayout.addWidget(self.browseForImagesPushbutton)


        self.verticalLayout_2.addLayout(self.rawImagesHorizontalLayout)

        self.outputDirectoryHorizontalLayout = QHBoxLayout()
        self.outputDirectoryHorizontalLayout.setObjectName(u"outputDirectoryHorizontalLayout")
        self.label_6 = QLabel(self.frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMinimumSize(QSize(0, 0))
        self.label_6.setMaximumSize(QSize(120, 16777215))
        self.label_6.setMargin(1)
        self.label_6.setIndent(5)

        self.outputDirectoryHorizontalLayout.addWidget(self.label_6)

        self.outputDirectoryLineEdit = QLineEdit(self.frame)
        self.outputDirectoryLineEdit.setObjectName(u"outputDirectoryLineEdit")
        self.outputDirectoryLineEdit.setMaximumSize(QSize(250, 16777215))

        self.outputDirectoryHorizontalLayout.addWidget(self.outputDirectoryLineEdit)

        self.browseoutputDirectoryPushbutton = QPushButton(self.frame)
        self.browseoutputDirectoryPushbutton.setObjectName(u"browseoutputDirectoryPushbutton")
        self.browseoutputDirectoryPushbutton.setMinimumSize(QSize(60, 0))
        self.browseoutputDirectoryPushbutton.setMaximumSize(QSize(100, 16777215))

        self.outputDirectoryHorizontalLayout.addWidget(self.browseoutputDirectoryPushbutton)


        self.verticalLayout_2.addLayout(self.outputDirectoryHorizontalLayout)

        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label)

        self.processingStatusBarFrame = QFrame(self.frame)
        self.processingStatusBarFrame.setObjectName(u"processingStatusBarFrame")
        self.processingStatusBarFrame.setFrameShape(QFrame.StyledPanel)
        self.processingStatusBarFrame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.processingStatusBarFrame)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.processingStatusProgressBar = QProgressBar(self.processingStatusBarFrame)
        self.processingStatusProgressBar.setObjectName(u"processingStatusProgressBar")
        self.processingStatusProgressBar.setMaximumSize(QSize(16777215, 10))
        font1 = QFont()
        font1.setPointSize(7)
        self.processingStatusProgressBar.setFont(font1)
        self.processingStatusProgressBar.setValue(24)
        self.processingStatusProgressBar.setOrientation(Qt.Horizontal)

        self.horizontalLayout_8.addWidget(self.processingStatusProgressBar)


        self.verticalLayout_2.addWidget(self.processingStatusBarFrame)

        self.imagesListWidget = QListWidget(self.frame)
        self.imagesListWidget.setObjectName(u"imagesListWidget")
        self.imagesListWidget.setMaximumSize(QSize(340, 16777215))

        self.verticalLayout_2.addWidget(self.imagesListWidget)

        self.thumbnailContainerScrollArea = QScrollArea(self.frame)
        self.thumbnailContainerScrollArea.setObjectName(u"thumbnailContainerScrollArea")
        self.thumbnailContainerScrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 344, 310))
        self.thumbnailContainerScrollArea.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_2.addWidget(self.thumbnailContainerScrollArea)

        self.frame_3 = QFrame(self.frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.displayImagesAsThumbnailsCheckBox = QCheckBox(self.frame_3)
        self.displayImagesAsThumbnailsCheckBox.setObjectName(u"displayImagesAsThumbnailsCheckBox")

        self.horizontalLayout_9.addWidget(self.displayImagesAsThumbnailsCheckBox)

        self.thumbnailZoomLevelHorizontalSlider = QSlider(self.frame_3)
        self.thumbnailZoomLevelHorizontalSlider.setObjectName(u"thumbnailZoomLevelHorizontalSlider")
        self.thumbnailZoomLevelHorizontalSlider.setMinimum(1)
        self.thumbnailZoomLevelHorizontalSlider.setMaximum(6)
        self.thumbnailZoomLevelHorizontalSlider.setValue(3)
        self.thumbnailZoomLevelHorizontalSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_9.addWidget(self.thumbnailZoomLevelHorizontalSlider)


        self.verticalLayout_2.addWidget(self.frame_3)


        self.verticalLayout_3.addLayout(self.verticalLayout_2)

        self.cpuUsageProgressBar = QProgressBar(self.frame)
        self.cpuUsageProgressBar.setObjectName(u"cpuUsageProgressBar")
        self.cpuUsageProgressBar.setMaximumSize(QSize(16777215, 10))
        self.cpuUsageProgressBar.setFont(font1)
        self.cpuUsageProgressBar.setValue(24)

        self.verticalLayout_3.addWidget(self.cpuUsageProgressBar)

        self.memoryUsageProgressBar = QProgressBar(self.frame)
        self.memoryUsageProgressBar.setObjectName(u"memoryUsageProgressBar")
        self.memoryUsageProgressBar.setMaximumSize(QSize(16777215, 10))
        self.memoryUsageProgressBar.setFont(font1)
        self.memoryUsageProgressBar.setValue(24)

        self.verticalLayout_3.addWidget(self.memoryUsageProgressBar)


        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)

        self.toolBox = QToolBox(self.centralwidget)
        self.toolBox.setObjectName(u"toolBox")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMinimumSize(QSize(240, 0))
        self.toolBox.setMaximumSize(QSize(240, 16777215))
        self.toolBox.setFrameShape(QFrame.NoFrame)
        self.colourChartToolsTab = QWidget()
        self.colourChartToolsTab.setObjectName(u"colourChartToolsTab")
        self.colourChartToolsTab.setGeometry(QRect(0, 0, 240, 669))
        self.formLayout_6 = QFormLayout(self.colourChartToolsTab)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.supportedColourChartsComboBox = QComboBox(self.colourChartToolsTab)
        self.supportedColourChartsComboBox.setObjectName(u"supportedColourChartsComboBox")

        self.formLayout_6.setWidget(1, QFormLayout.SpanningRole, self.supportedColourChartsComboBox)

        self.setSelectedAsChartPushbutton = QPushButton(self.colourChartToolsTab)
        self.setSelectedAsChartPushbutton.setObjectName(u"setSelectedAsChartPushbutton")

        self.formLayout_6.setWidget(2, QFormLayout.SpanningRole, self.setSelectedAsChartPushbutton)

        self.manuallySelectChartPushbutton = QPushButton(self.colourChartToolsTab)
        self.manuallySelectChartPushbutton.setObjectName(u"manuallySelectChartPushbutton")

        self.formLayout_6.setWidget(3, QFormLayout.SpanningRole, self.manuallySelectChartPushbutton)

        self.exportChartConfigPushButton = QPushButton(self.colourChartToolsTab)
        self.exportChartConfigPushButton.setObjectName(u"exportChartConfigPushButton")

        self.formLayout_6.setWidget(4, QFormLayout.SpanningRole, self.exportChartConfigPushButton)

        self.line_2 = QFrame(self.colourChartToolsTab)
        self.line_2.setObjectName(u"line_2")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.line_2.sizePolicy().hasHeightForWidth())
        self.line_2.setSizePolicy(sizePolicy1)
        self.line_2.setMinimumSize(QSize(100, 0))
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.formLayout_6.setWidget(5, QFormLayout.SpanningRole, self.line_2)

        self.label_8 = QLabel(self.colourChartToolsTab)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMinimumSize(QSize(0, 0))
        self.label_8.setMaximumSize(QSize(120, 16777215))
        self.label_8.setMargin(1)
        self.label_8.setIndent(5)

        self.formLayout_6.setWidget(6, QFormLayout.LabelRole, self.label_8)

        self.chartPathLineEdit = QLineEdit(self.colourChartToolsTab)
        self.chartPathLineEdit.setObjectName(u"chartPathLineEdit")
        self.chartPathLineEdit.setMaximumSize(QSize(250, 16777215))

        self.formLayout_6.setWidget(6, QFormLayout.FieldRole, self.chartPathLineEdit)

        self.browseForChartPushbutton = QPushButton(self.colourChartToolsTab)
        self.browseForChartPushbutton.setObjectName(u"browseForChartPushbutton")
        self.browseForChartPushbutton.setMinimumSize(QSize(60, 0))
        self.browseForChartPushbutton.setMaximumSize(QSize(100, 16777215))

        self.formLayout_6.setWidget(7, QFormLayout.FieldRole, self.browseForChartPushbutton)

        self.groupBox_3 = QGroupBox(self.colourChartToolsTab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_7 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.usePrecalcChartCheckbox = QCheckBox(self.groupBox_3)
        self.usePrecalcChartCheckbox.setObjectName(u"usePrecalcChartCheckbox")
        self.usePrecalcChartCheckbox.setLayoutDirection(Qt.RightToLeft)

        self.verticalLayout_7.addWidget(self.usePrecalcChartCheckbox)

        self.precalcChartComboBox = QComboBox(self.groupBox_3)
        self.precalcChartComboBox.setObjectName(u"precalcChartComboBox")

        self.verticalLayout_7.addWidget(self.precalcChartComboBox)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.verticalLayout_7.addItem(self.verticalSpacer_4)

        self.line_3 = QFrame(self.groupBox_3)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_3.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_7.addWidget(self.line_3)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.verticalLayout_7.addItem(self.verticalSpacer_3)

        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_3)

        self.useSelectedChartForAllGroupsPushbutton = QPushButton(self.groupBox_3)
        self.useSelectedChartForAllGroupsPushbutton.setObjectName(u"useSelectedChartForAllGroupsPushbutton")

        self.verticalLayout_7.addWidget(self.useSelectedChartForAllGroupsPushbutton)

        self.removeSelectedChartPushbutton = QPushButton(self.groupBox_3)
        self.removeSelectedChartPushbutton.setObjectName(u"removeSelectedChartPushbutton")

        self.verticalLayout_7.addWidget(self.removeSelectedChartPushbutton)

        self.copyChartPushbutton = QPushButton(self.groupBox_3)
        self.copyChartPushbutton.setObjectName(u"copyChartPushbutton")

        self.verticalLayout_7.addWidget(self.copyChartPushbutton)

        self.pasteChartPushbutton = QPushButton(self.groupBox_3)
        self.pasteChartPushbutton.setObjectName(u"pasteChartPushbutton")

        self.verticalLayout_7.addWidget(self.pasteChartPushbutton)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_7.addItem(self.verticalSpacer_2)


        self.formLayout_6.setWidget(8, QFormLayout.SpanningRole, self.groupBox_3)

        self.dontUseColourChartCheckBox = QCheckBox(self.colourChartToolsTab)
        self.dontUseColourChartCheckBox.setObjectName(u"dontUseColourChartCheckBox")
        self.dontUseColourChartCheckBox.setLayoutDirection(Qt.RightToLeft)

        self.formLayout_6.setWidget(0, QFormLayout.SpanningRole, self.dontUseColourChartCheckBox)

        self.toolBox.addItem(self.colourChartToolsTab, u"Colour Chart Tools")
        self.averageExposureToolsTab = QWidget()
        self.averageExposureToolsTab.setObjectName(u"averageExposureToolsTab")
        self.averageExposureToolsTab.setGeometry(QRect(0, 0, 240, 669))
        self.formLayout_3 = QFormLayout(self.averageExposureToolsTab)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.calculateAverageExposurePushbutton = QPushButton(self.averageExposureToolsTab)
        self.calculateAverageExposurePushbutton.setObjectName(u"calculateAverageExposurePushbutton")

        self.formLayout_3.setWidget(0, QFormLayout.SpanningRole, self.calculateAverageExposurePushbutton)

        self.setSelectedImageAsAveragePushbutton = QPushButton(self.averageExposureToolsTab)
        self.setSelectedImageAsAveragePushbutton.setObjectName(u"setSelectedImageAsAveragePushbutton")

        self.formLayout_3.setWidget(1, QFormLayout.SpanningRole, self.setSelectedImageAsAveragePushbutton)

        self.removeAverageDataPushbutton = QPushButton(self.averageExposureToolsTab)
        self.removeAverageDataPushbutton.setObjectName(u"removeAverageDataPushbutton")

        self.formLayout_3.setWidget(2, QFormLayout.SpanningRole, self.removeAverageDataPushbutton)

        self.highlightLimitSpinBox = QSpinBox(self.averageExposureToolsTab)
        self.highlightLimitSpinBox.setObjectName(u"highlightLimitSpinBox")
        self.highlightLimitSpinBox.setValue(96)

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.highlightLimitSpinBox)

        self.label_13 = QLabel(self.averageExposureToolsTab)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setIndent(9)

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.label_13)

        self.shadowLimitSpinBox = QSpinBox(self.averageExposureToolsTab)
        self.shadowLimitSpinBox.setObjectName(u"shadowLimitSpinBox")
        self.shadowLimitSpinBox.setValue(6)

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.shadowLimitSpinBox)

        self.groupBox_2 = QGroupBox(self.averageExposureToolsTab)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout = QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.displayDebugExposureDataCheckBox = QCheckBox(self.groupBox_2)
        self.displayDebugExposureDataCheckBox.setObjectName(u"displayDebugExposureDataCheckBox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.displayDebugExposureDataCheckBox.sizePolicy().hasHeightForWidth())
        self.displayDebugExposureDataCheckBox.setSizePolicy(sizePolicy2)
        self.displayDebugExposureDataCheckBox.setSizeIncrement(QSize(0, 0))
        font2 = QFont()
        font2.setPointSize(9)
        self.displayDebugExposureDataCheckBox.setFont(font2)

        self.verticalLayout.addWidget(self.displayDebugExposureDataCheckBox)

        self.exportMaskedImagesCheckBox = QCheckBox(self.groupBox_2)
        self.exportMaskedImagesCheckBox.setObjectName(u"exportMaskedImagesCheckBox")

        self.verticalLayout.addWidget(self.exportMaskedImagesCheckBox)


        self.formLayout_3.setWidget(5, QFormLayout.SpanningRole, self.groupBox_2)

        self.label_12 = QLabel(self.averageExposureToolsTab)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setIndent(9)

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.label_12)

        self.toolBox.addItem(self.averageExposureToolsTab, u"Average Exposure Tools")
        self.imageNamingTab = QWidget()
        self.imageNamingTab.setObjectName(u"imageNamingTab")
        self.formLayout_4 = QFormLayout(self.imageNamingTab)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.useOriginalFilenamesCheckBox = QCheckBox(self.imageNamingTab)
        self.useOriginalFilenamesCheckBox.setObjectName(u"useOriginalFilenamesCheckBox")
        self.useOriginalFilenamesCheckBox.setLayoutDirection(Qt.RightToLeft)

        self.formLayout_4.setWidget(0, QFormLayout.SpanningRole, self.useOriginalFilenamesCheckBox)

        self.label_5 = QLabel(self.imageNamingTab)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_4.setWidget(1, QFormLayout.LabelRole, self.label_5)

        self.newImageNameLineEdit = QLineEdit(self.imageNamingTab)
        self.newImageNameLineEdit.setObjectName(u"newImageNameLineEdit")

        self.formLayout_4.setWidget(1, QFormLayout.FieldRole, self.newImageNameLineEdit)

        self.label_7 = QLabel(self.imageNamingTab)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setIndent(8)

        self.formLayout_4.setWidget(2, QFormLayout.LabelRole, self.label_7)

        self.imagePaddingSpinBox = QSpinBox(self.imageNamingTab)
        self.imagePaddingSpinBox.setObjectName(u"imagePaddingSpinBox")
        self.imagePaddingSpinBox.setMaximum(6)
        self.imagePaddingSpinBox.setValue(4)

        self.formLayout_4.setWidget(2, QFormLayout.FieldRole, self.imagePaddingSpinBox)

        self.groupBox_4 = QGroupBox(self.imageNamingTab)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setMaximumSize(QSize(220, 300))
        self.verticalLayout_8 = QVBoxLayout(self.groupBox_4)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.schemaNameLineEdit = QLineEdit(self.groupBox_4)
        self.schemaNameLineEdit.setObjectName(u"schemaNameLineEdit")

        self.verticalLayout_8.addWidget(self.schemaNameLineEdit)

        self.schemaInfoQTextEdit = QTextEdit(self.groupBox_4)
        self.schemaInfoQTextEdit.setObjectName(u"schemaInfoQTextEdit")
        self.schemaInfoQTextEdit.setMaximumSize(QSize(16777215, 200))
        self.schemaInfoQTextEdit.setReadOnly(True)

        self.verticalLayout_8.addWidget(self.schemaInfoQTextEdit)

        self.label_21 = QLabel(self.groupBox_4)
        self.label_21.setObjectName(u"label_21")

        self.verticalLayout_8.addWidget(self.label_21)

        self.schemaOutputPathForSelectedLabel = QLabel(self.groupBox_4)
        self.schemaOutputPathForSelectedLabel.setObjectName(u"schemaOutputPathForSelectedLabel")

        self.verticalLayout_8.addWidget(self.schemaOutputPathForSelectedLabel)


        self.formLayout_4.setWidget(3, QFormLayout.SpanningRole, self.groupBox_4)

        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.formLayout_4.setItem(4, QFormLayout.SpanningRole, self.verticalSpacer_5)

        self.verticalSpacer_6 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.formLayout_4.setItem(5, QFormLayout.SpanningRole, self.verticalSpacer_6)

        self.toolBox.addItem(self.imageNamingTab, u"Image Naming")
        self.imageEditingTab = QWidget()
        self.imageEditingTab.setObjectName(u"imageEditingTab")
        self.imageEditingTab.setMaximumSize(QSize(240, 16777215))
        self.gridLayout = QGridLayout(self.imageEditingTab)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_17 = QLabel(self.imageEditingTab)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setMaximumSize(QSize(16777215, 20))
        font3 = QFont()
        font3.setPointSize(8)
        self.label_17.setFont(font3)

        self.gridLayout.addWidget(self.label_17, 5, 0, 1, 2)

        self.shadowAdjustmentDoubleSpinBox = QDoubleSpinBox(self.imageEditingTab)
        self.shadowAdjustmentDoubleSpinBox.setObjectName(u"shadowAdjustmentDoubleSpinBox")
        self.shadowAdjustmentDoubleSpinBox.setMinimum(-1.000000000000000)
        self.shadowAdjustmentDoubleSpinBox.setMaximum(1.000000000000000)
        self.shadowAdjustmentDoubleSpinBox.setSingleStep(0.050000000000000)

        self.gridLayout.addWidget(self.shadowAdjustmentDoubleSpinBox, 6, 1, 1, 1)

        self.sharpenHorizontalSlider = QSlider(self.imageEditingTab)
        self.sharpenHorizontalSlider.setObjectName(u"sharpenHorizontalSlider")
        self.sharpenHorizontalSlider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.sharpenHorizontalSlider, 13, 0, 1, 2)

        self.resetEditSettingsPushButton = QPushButton(self.imageEditingTab)
        self.resetEditSettingsPushButton.setObjectName(u"resetEditSettingsPushButton")

        self.gridLayout.addWidget(self.resetEditSettingsPushButton, 17, 0, 1, 2)

        self.exposureAdjustmentSlider = QSlider(self.imageEditingTab)
        self.exposureAdjustmentSlider.setObjectName(u"exposureAdjustmentSlider")
        self.exposureAdjustmentSlider.setMinimum(-50)
        self.exposureAdjustmentSlider.setMaximum(50)
        self.exposureAdjustmentSlider.setSingleStep(1)
        self.exposureAdjustmentSlider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.exposureAdjustmentSlider, 4, 0, 1, 1)

        self.sharpenImageCheckBox = QCheckBox(self.imageEditingTab)
        self.sharpenImageCheckBox.setObjectName(u"sharpenImageCheckBox")

        self.gridLayout.addWidget(self.sharpenImageCheckBox, 12, 0, 1, 1)

        self.highlightAdjustmentDoubleSpinBox = QDoubleSpinBox(self.imageEditingTab)
        self.highlightAdjustmentDoubleSpinBox.setObjectName(u"highlightAdjustmentDoubleSpinBox")
        self.highlightAdjustmentDoubleSpinBox.setMinimum(-1.000000000000000)
        self.highlightAdjustmentDoubleSpinBox.setMaximum(1.000000000000000)
        self.highlightAdjustmentDoubleSpinBox.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.highlightAdjustmentDoubleSpinBox, 8, 1, 1, 1)

        self.sharpenDoubleSpinBox = QDoubleSpinBox(self.imageEditingTab)
        self.sharpenDoubleSpinBox.setObjectName(u"sharpenDoubleSpinBox")

        self.gridLayout.addWidget(self.sharpenDoubleSpinBox, 12, 1, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 18, 0, 1, 2)

        self.groupBox = QGroupBox(self.imageEditingTab)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout_3 = QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.whitebalanceSpinbox = QSpinBox(self.groupBox)
        self.whitebalanceSpinbox.setObjectName(u"whitebalanceSpinbox")
        self.whitebalanceSpinbox.setButtonSymbols(QAbstractSpinBox.UpDownArrows)
        self.whitebalanceSpinbox.setMinimum(1000)
        self.whitebalanceSpinbox.setMaximum(14000)
        self.whitebalanceSpinbox.setSingleStep(50)
        self.whitebalanceSpinbox.setValue(5500)

        self.gridLayout_3.addWidget(self.whitebalanceSpinbox, 1, 1, 1, 1)

        self.sampleWhiteBalancePushButton = QPushButton(self.groupBox)
        self.sampleWhiteBalancePushButton.setObjectName(u"sampleWhiteBalancePushButton")

        self.gridLayout_3.addWidget(self.sampleWhiteBalancePushButton, 1, 0, 1, 1)

        self.enableWhiteBalanceCheckBox = QCheckBox(self.groupBox)
        self.enableWhiteBalanceCheckBox.setObjectName(u"enableWhiteBalanceCheckBox")

        self.gridLayout_3.addWidget(self.enableWhiteBalanceCheckBox, 0, 0, 1, 2)


        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 2)

        self.label_10 = QLabel(self.imageEditingTab)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setMaximumSize(QSize(16777215, 20))
        font4 = QFont()
        font4.setPointSize(8)
        font4.setBold(False)
        self.label_10.setFont(font4)

        self.gridLayout.addWidget(self.label_10, 3, 0, 1, 2)

        self.denoiseDoubleSpinBox = QDoubleSpinBox(self.imageEditingTab)
        self.denoiseDoubleSpinBox.setObjectName(u"denoiseDoubleSpinBox")

        self.gridLayout.addWidget(self.denoiseDoubleSpinBox, 9, 1, 1, 1)

        self.shadowAdjustmentSlider = QSlider(self.imageEditingTab)
        self.shadowAdjustmentSlider.setObjectName(u"shadowAdjustmentSlider")
        self.shadowAdjustmentSlider.setMinimum(-1000)
        self.shadowAdjustmentSlider.setMaximum(1000)
        self.shadowAdjustmentSlider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.shadowAdjustmentSlider, 6, 0, 1, 1)

        self.denoiseImageCheckBox = QCheckBox(self.imageEditingTab)
        self.denoiseImageCheckBox.setObjectName(u"denoiseImageCheckBox")

        self.gridLayout.addWidget(self.denoiseImageCheckBox, 9, 0, 1, 1)

        self.line = QFrame(self.imageEditingTab)
        self.line.setObjectName(u"line")
        sizePolicy1.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy1)
        self.line.setMinimumSize(QSize(150, 0))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 14, 0, 1, 2)

        self.label_18 = QLabel(self.imageEditingTab)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setMaximumSize(QSize(16777215, 30))
        self.label_18.setFont(font3)

        self.gridLayout.addWidget(self.label_18, 7, 0, 1, 2)

        self.highlightAdjustmentSlider = QSlider(self.imageEditingTab)
        self.highlightAdjustmentSlider.setObjectName(u"highlightAdjustmentSlider")
        self.highlightAdjustmentSlider.setMinimum(-1000)
        self.highlightAdjustmentSlider.setMaximum(1000)
        self.highlightAdjustmentSlider.setValue(0)
        self.highlightAdjustmentSlider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.highlightAdjustmentSlider, 8, 0, 1, 1)

        self.exposureAdjustmentDoubleSpinBox = QDoubleSpinBox(self.imageEditingTab)
        self.exposureAdjustmentDoubleSpinBox.setObjectName(u"exposureAdjustmentDoubleSpinBox")
        self.exposureAdjustmentDoubleSpinBox.setMinimum(-5.000000000000000)
        self.exposureAdjustmentDoubleSpinBox.setMaximum(5.000000000000000)
        self.exposureAdjustmentDoubleSpinBox.setSingleStep(0.250000000000000)

        self.gridLayout.addWidget(self.exposureAdjustmentDoubleSpinBox, 4, 1, 1, 1)

        self.denoiseHorizontalSlider = QSlider(self.imageEditingTab)
        self.denoiseHorizontalSlider.setObjectName(u"denoiseHorizontalSlider")
        self.denoiseHorizontalSlider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.denoiseHorizontalSlider, 10, 0, 1, 2)

        self.groupBox_5 = QGroupBox(self.imageEditingTab)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.gridLayout_4 = QGridLayout(self.groupBox_5)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_9 = QLabel(self.groupBox_5)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_4.addWidget(self.label_9, 2, 0, 1, 1)

        self.enableRescaleCheckBox = QCheckBox(self.groupBox_5)
        self.enableRescaleCheckBox.setObjectName(u"enableRescaleCheckBox")

        self.gridLayout_4.addWidget(self.enableRescaleCheckBox, 0, 0, 1, 1)

        self.frame_2 = QFrame(self.groupBox_5)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.xDimensionOutputSpinbox = QSpinBox(self.frame_2)
        self.xDimensionOutputSpinbox.setObjectName(u"xDimensionOutputSpinbox")

        self.horizontalLayout_7.addWidget(self.xDimensionOutputSpinbox)

        self.label_19 = QLabel(self.frame_2)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_7.addWidget(self.label_19)

        self.yDiemnsionOutputSpinbox = QSpinBox(self.frame_2)
        self.yDiemnsionOutputSpinbox.setObjectName(u"yDiemnsionOutputSpinbox")

        self.horizontalLayout_7.addWidget(self.yDiemnsionOutputSpinbox)


        self.gridLayout_4.addWidget(self.frame_2, 4, 0, 1, 3)

        self.targetDownsamplePercentDoubleSpinbox = QDoubleSpinBox(self.groupBox_5)
        self.targetDownsamplePercentDoubleSpinbox.setObjectName(u"targetDownsamplePercentDoubleSpinbox")
        self.targetDownsamplePercentDoubleSpinbox.setMaximum(100.000000000000000)
        self.targetDownsamplePercentDoubleSpinbox.setValue(100.000000000000000)

        self.gridLayout_4.addWidget(self.targetDownsamplePercentDoubleSpinbox, 1, 2, 1, 1)

        self.label_11 = QLabel(self.groupBox_5)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)

        self.targetDownsampleMpxDoubleSpinBox = QDoubleSpinBox(self.groupBox_5)
        self.targetDownsampleMpxDoubleSpinBox.setObjectName(u"targetDownsampleMpxDoubleSpinBox")

        self.gridLayout_4.addWidget(self.targetDownsampleMpxDoubleSpinBox, 2, 2, 1, 1)

        self.label_20 = QLabel(self.groupBox_5)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setAlignment(Qt.AlignCenter)

        self.gridLayout_4.addWidget(self.label_20, 3, 0, 1, 3)


        self.gridLayout.addWidget(self.groupBox_5, 0, 0, 1, 2)

        self.toolBox.addItem(self.imageEditingTab, u"Image Editing")
        self.processExportTab = QWidget()
        self.processExportTab.setObjectName(u"processExportTab")
        self.formLayout_7 = QFormLayout(self.processExportTab)
        self.formLayout_7.setObjectName(u"formLayout_7")
        self.label_16 = QLabel(self.processExportTab)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setIndent(9)

        self.formLayout_7.setWidget(0, QFormLayout.LabelRole, self.label_16)

        self.imageFormatComboBox = QComboBox(self.processExportTab)
        self.imageFormatComboBox.setObjectName(u"imageFormatComboBox")

        self.formLayout_7.setWidget(0, QFormLayout.FieldRole, self.imageFormatComboBox)

        self.bitDepthFrame = QGroupBox(self.processExportTab)
        self.bitDepthFrame.setObjectName(u"bitDepthFrame")
        self.horizontalLayout_4 = QHBoxLayout(self.bitDepthFrame)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.eightBitRadioButton = QRadioButton(self.bitDepthFrame)
        self.eightBitRadioButton.setObjectName(u"eightBitRadioButton")

        self.horizontalLayout_4.addWidget(self.eightBitRadioButton)

        self.sixteenBitRadioButton = QRadioButton(self.bitDepthFrame)
        self.sixteenBitRadioButton.setObjectName(u"sixteenBitRadioButton")

        self.horizontalLayout_4.addWidget(self.sixteenBitRadioButton)


        self.formLayout_7.setWidget(1, QFormLayout.SpanningRole, self.bitDepthFrame)

        self.imageProcessingThreadsSpinbox = QSpinBox(self.processExportTab)
        self.imageProcessingThreadsSpinbox.setObjectName(u"imageProcessingThreadsSpinbox")
        self.imageProcessingThreadsSpinbox.setMinimum(1)
        self.imageProcessingThreadsSpinbox.setValue(4)

        self.formLayout_7.setWidget(4, QFormLayout.FieldRole, self.imageProcessingThreadsSpinbox)

        self.processImagesPushbutton = QPushButton(self.processExportTab)
        self.processImagesPushbutton.setObjectName(u"processImagesPushbutton")
        self.processImagesPushbutton.setMinimumSize(QSize(100, 40))

        self.formLayout_7.setWidget(7, QFormLayout.SpanningRole, self.processImagesPushbutton)

        self.exportCurrentProjectPushButton = QPushButton(self.processExportTab)
        self.exportCurrentProjectPushButton.setObjectName(u"exportCurrentProjectPushButton")

        self.formLayout_7.setWidget(8, QFormLayout.SpanningRole, self.exportCurrentProjectPushButton)

        self.sendProjectToServerPushButton = QPushButton(self.processExportTab)
        self.sendProjectToServerPushButton.setObjectName(u"sendProjectToServerPushButton")

        self.formLayout_7.setWidget(9, QFormLayout.SpanningRole, self.sendProjectToServerPushButton)

        self.label_4 = QLabel(self.processExportTab)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMargin(5)
        self.label_4.setIndent(5)

        self.formLayout_7.setWidget(4, QFormLayout.LabelRole, self.label_4)

        self.jpgQualityFrame = QGroupBox(self.processExportTab)
        self.jpgQualityFrame.setObjectName(u"jpgQualityFrame")
        self.horizontalLayout_2 = QHBoxLayout(self.jpgQualityFrame)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.jpegQualitySpinbox_2 = QSpinBox(self.jpgQualityFrame)
        self.jpegQualitySpinbox_2.setObjectName(u"jpegQualitySpinbox_2")
        self.jpegQualitySpinbox_2.setMinimum(10)
        self.jpegQualitySpinbox_2.setMaximum(100)
        self.jpegQualitySpinbox_2.setSingleStep(10)
        self.jpegQualitySpinbox_2.setValue(100)

        self.horizontalLayout_2.addWidget(self.jpegQualitySpinbox_2)

        self.label_14 = QLabel(self.jpgQualityFrame)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setMinimumSize(QSize(80, 0))
        self.label_14.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_14.setMargin(0)
        self.label_14.setIndent(9)

        self.horizontalLayout_2.addWidget(self.label_14)


        self.formLayout_7.setWidget(3, QFormLayout.SpanningRole, self.jpgQualityFrame)

        self.exrOptionsFrame = QGroupBox(self.processExportTab)
        self.exrOptionsFrame.setObjectName(u"exrOptionsFrame")
        self.horizontalLayout_6 = QHBoxLayout(self.exrOptionsFrame)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_15 = QLabel(self.exrOptionsFrame)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setIndent(9)

        self.horizontalLayout_6.addWidget(self.label_15)

        self.exrColourSpaceComboBox = QComboBox(self.exrOptionsFrame)
        self.exrColourSpaceComboBox.setObjectName(u"exrColourSpaceComboBox")

        self.horizontalLayout_6.addWidget(self.exrColourSpaceComboBox)


        self.formLayout_7.setWidget(2, QFormLayout.SpanningRole, self.exrOptionsFrame)

        self.toolBox.addItem(self.processExportTab, u"Process and Export")

        self.gridLayout_2.addWidget(self.toolBox, 0, 1, 1, 2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 1471, 22))
        self.menuFile = QMenu(self.menuBar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuHelp = QMenu(self.menuBar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionLoadSettingsMenu)
        self.menuFile.addAction(self.actionQuit)
        self.menuHelp.addAction(self.actionDocumentation)
        self.menuHelp.addAction(self.actionGithub)

        self.retranslateUi(MainWindow)

        self.toolBox.setCurrentIndex(0)
        self.toolBox.layout().setSpacing(6)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Image Space", None))
        self.actionFile.setText(QCoreApplication.translate("MainWindow", u"File", None))
        self.actionDocumentation.setText(QCoreApplication.translate("MainWindow", u"Documentation", None))
        self.actionGithub.setText(QCoreApplication.translate("MainWindow", u"Github", None))
        self.actionSettings.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.actionClient.setText(QCoreApplication.translate("MainWindow", u"Client", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
        self.actionLoadSettingsMenu.setText(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.actionImport_Folder.setText(QCoreApplication.translate("MainWindow", u"Import Folder", None))
        self.actionVersion_1.setText(QCoreApplication.translate("MainWindow", u"Version 1", None))
        self.actionVersion_2.setText(QCoreApplication.translate("MainWindow", u"Version 2", None))
        self.actionNetwork_Processing_Monitor.setText(QCoreApplication.translate("MainWindow", u"Network Processing Monitor", None))
        self.actionBatch_Processor.setText(QCoreApplication.translate("MainWindow", u"Batch Processor", None))
        self.serverStatusLabel.setText(QCoreApplication.translate("MainWindow", u"Server Satus:", None))
        self.showOriginalImagePushbutton.setText(QCoreApplication.translate("MainWindow", u"Show Original Image", None))
#if QT_CONFIG(tooltip)
        self.flattenChartImagePushButton.setToolTip(QCoreApplication.translate("MainWindow", u"Select the four corners of the chart to straighten it.", None))
#endif // QT_CONFIG(tooltip)
        self.flattenChartImagePushButton.setText(QCoreApplication.translate("MainWindow", u"Flatten Chart Image", None))
        self.revertImagePushbutton.setText(QCoreApplication.translate("MainWindow", u"Revert Image", None))
        self.detectChartShelfPushbutton.setText(QCoreApplication.translate("MainWindow", u"Detect Chart", None))
        self.finalizeChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Finalize Chart", None))
        self.chartInformationLabel.setText("")
        self.correctedImageRadioButton.setText(QCoreApplication.translate("MainWindow", u"Corrected Image", None))
        self.swatchOverlayRadioButton.setText(QCoreApplication.translate("MainWindow", u"Swatch Overlay", None))
        self.detectionDebugRadioButton.setText(QCoreApplication.translate("MainWindow", u"Detection Debug", None))
        self.swatchAndClusterRadioButton.setText(QCoreApplication.translate("MainWindow", u"Swatches And Clusters", None))
        self.previousImagePushbutton.setText(QCoreApplication.translate("MainWindow", u"<", None))
        self.nextImagePushbutton.setText(QCoreApplication.translate("MainWindow", u">", None))
        self.logOutputTextEdit.setDocumentTitle(QCoreApplication.translate("MainWindow", u"Log", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Raw Image Directory", None))
        self.browseForImagesPushbutton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Output Directory", None))
        self.browseoutputDirectoryPushbutton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Images", None))
        self.displayImagesAsThumbnailsCheckBox.setText(QCoreApplication.translate("MainWindow", u"Display Thumbnails", None))
        self.supportedColourChartsComboBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Supported Colour Charts", None))
        self.setSelectedAsChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Set Selected As Chart", None))
        self.manuallySelectChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Manually Select Chart", None))
        self.exportChartConfigPushButton.setText(QCoreApplication.translate("MainWindow", u"Export Chart Config", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Chart Image", None))
        self.browseForChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Chart Assignment", None))
        self.usePrecalcChartCheckbox.setText(QCoreApplication.translate("MainWindow", u"Enable Chart Override", None))
#if QT_CONFIG(tooltip)
        self.precalcChartComboBox.setToolTip(QCoreApplication.translate("MainWindow", u"Set your custom chart folder in settings.", None))
#endif // QT_CONFIG(tooltip)
        self.precalcChartComboBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Precalculated Chart", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Advanced", None))
        self.useSelectedChartForAllGroupsPushbutton.setText(QCoreApplication.translate("MainWindow", u"Use Slected Chart for All Groups", None))
        self.removeSelectedChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Remove Selected Chart", None))
        self.copyChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Copy Chart from Group", None))
        self.pasteChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Paste Chart to Group", None))
        self.dontUseColourChartCheckBox.setText(QCoreApplication.translate("MainWindow", u"Don't Use Colour Chart", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.colourChartToolsTab), QCoreApplication.translate("MainWindow", u"Colour Chart Tools", None))
        self.calculateAverageExposurePushbutton.setText(QCoreApplication.translate("MainWindow", u"Calculate Average Exposure", None))
        self.setSelectedImageAsAveragePushbutton.setText(QCoreApplication.translate("MainWindow", u"Set Selected image as Average", None))
        self.removeAverageDataPushbutton.setText(QCoreApplication.translate("MainWindow", u"Remove Average Data", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Shadow Limit", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Options", None))
#if QT_CONFIG(tooltip)
        self.displayDebugExposureDataCheckBox.setToolTip(QCoreApplication.translate("MainWindow", u"Display overlays over each image showing the hot and cold spots of each image", None))
#endif // QT_CONFIG(tooltip)
        self.displayDebugExposureDataCheckBox.setText(QCoreApplication.translate("MainWindow", u"Display debug data", None))
        self.exportMaskedImagesCheckBox.setText(QCoreApplication.translate("MainWindow", u"Export Masked Images", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Highlight Limit", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.averageExposureToolsTab), QCoreApplication.translate("MainWindow", u"Average Exposure Tools", None))
        self.useOriginalFilenamesCheckBox.setText(QCoreApplication.translate("MainWindow", u"Use Original Filenames", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Image Prefix", None))
#if QT_CONFIG(tooltip)
        self.newImageNameLineEdit.setToolTip(QCoreApplication.translate("MainWindow", u"New Image Name, will be followed by _0001, numbers are based on padding length", None))
#endif // QT_CONFIG(tooltip)
        self.newImageNameLineEdit.setText(QCoreApplication.translate("MainWindow", u"Image", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Padding", None))
#if QT_CONFIG(tooltip)
        self.imagePaddingSpinBox.setToolTip(QCoreApplication.translate("MainWindow", u"Indicates how  many digits the image number will be", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.imagePaddingSpinBox.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"Schema Naming", None))
        self.schemaNameLineEdit.setText(QCoreApplication.translate("MainWindow", u"[r]/[o][n4][e]", None))
        self.schemaInfoQTextEdit.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[r]</span>: Root folder name</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[s]</span>: Sub folder name (if exists)</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span sty"
                        "le=\" font-weight:700;\">[e]</span>: File name extension</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[oc]</span>: Original file name without numbers</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[c]</span>: Custom name from custom input</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[o]</span>: Original file name</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[n]</span>: Image Number</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">[/]</sp"
                        "an>: new folder layer</p></body></html>", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"Output Path For Selected:", None))
        self.schemaOutputPathForSelectedLabel.setText(QCoreApplication.translate("MainWindow", u"output_path", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.imageNamingTab), QCoreApplication.translate("MainWindow", u"Image Naming", None))
        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Shadows", None))
        self.resetEditSettingsPushButton.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.sharpenImageCheckBox.setText(QCoreApplication.translate("MainWindow", u"Sharpen", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"White Balance", None))
        self.whitebalanceSpinbox.setSuffix(QCoreApplication.translate("MainWindow", u"k", None))
        self.sampleWhiteBalancePushButton.setText(QCoreApplication.translate("MainWindow", u"Sample WB", None))
        self.enableWhiteBalanceCheckBox.setText(QCoreApplication.translate("MainWindow", u"Enable WB", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Exposure", None))
        self.denoiseImageCheckBox.setText(QCoreApplication.translate("MainWindow", u"Denoise", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Highlights", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Downsample Image", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Target Mpx", None))
        self.enableRescaleCheckBox.setText(QCoreApplication.translate("MainWindow", u"Enable Downsample", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"by", None))
        self.targetDownsamplePercentDoubleSpinbox.setSuffix(QCoreApplication.translate("MainWindow", u"%", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Target Percent", None))
        self.targetDownsampleMpxDoubleSpinBox.setSuffix(QCoreApplication.translate("MainWindow", u"mp", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Selected Images Output Dimensions", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.imageEditingTab), QCoreApplication.translate("MainWindow", u"Image Editing", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Format", None))
        self.bitDepthFrame.setTitle(QCoreApplication.translate("MainWindow", u"Bit Depth", None))
        self.eightBitRadioButton.setText(QCoreApplication.translate("MainWindow", u"8 bit", None))
        self.sixteenBitRadioButton.setText(QCoreApplication.translate("MainWindow", u"16 bit", None))
        self.processImagesPushbutton.setText(QCoreApplication.translate("MainWindow", u"Process Images", None))
        self.exportCurrentProjectPushButton.setText(QCoreApplication.translate("MainWindow", u"Export Current Project", None))
        self.sendProjectToServerPushButton.setText(QCoreApplication.translate("MainWindow", u"Send Project to Server", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Threads", None))
        self.jpgQualityFrame.setTitle(QCoreApplication.translate("MainWindow", u"Jpg Quality", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Jpeg Quality", None))
        self.exrOptionsFrame.setTitle(QCoreApplication.translate("MainWindow", u"Colourspace", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Color Space", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.processExportTab), QCoreApplication.translate("MainWindow", u"Process and Export", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi

