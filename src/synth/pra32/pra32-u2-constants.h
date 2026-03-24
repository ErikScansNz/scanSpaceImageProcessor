#pragma once

#include <cstdint>

const double    A4_FREQ = 440.0;

const uint32_t  SAMPLING_RATE       = 48000;
const uint32_t  FREQUENCY_MAX       = 23000;
const uint8_t   BIT_DEPTH           = 8;
const uint8_t   NOTE_NUMBER_MIN     = 0;
const uint8_t   NOTE_NUMBER_MAX     = 127;
const uint8_t   NOTE_NUMBER_INVALID = 255;

const uint8_t   USER_PROGRAM_NUMBER_MAX      = 7;
const uint8_t   PROGRAM_NUMBER_MAX           = 15;

const int8_t    OSC_CONTROL_INTERVAL_BITS    = 1;
const uint8_t   OSC_CONTROL_INTERVAL         = 0x01 << OSC_CONTROL_INTERVAL_BITS;
const int8_t    OSC_PHASE_RESOLUTION_BITS    = 24;
const int8_t    OSC_TUNE_TABLE_STEPS_BITS    = 8;
const int8_t    OSC_TUNE_DENOMINATOR_BITS    = 15;
const uint8_t   OSC_WAVE_TABLE_AMP_BITS      = 13;
const int16_t   OSC_WAVE_TABLE_AMP           = 1 << OSC_WAVE_TABLE_AMP_BITS;
const int8_t    OSC_WAVE_TABLE_SAMPLES_BITS  = 9;
const uint16_t  OSC_WAVE_TABLE_LAST_HARMONIC = 255;
const uint16_t  OSC_WAVE_SHAPE_TABLE_LEN_X   = 128 + 1 + 1;
const uint16_t  OSC_WAVE_SHAPE_TABLE_LEN_Y   = 16;
const int8_t    OSC_DETUNE_MUL_NUM_BITS      = 4;
const uint16_t  OSC_DETUNE_FREQ_MAX          = (10 * 2) * (1 << 24) / SAMPLING_RATE;
const int8_t    FILTER_CONTROL_INTERVAL_BITS = 3;
const uint8_t   FILTER_CONTROL_INTERVAL      = 0x01 << FILTER_CONTROL_INTERVAL_BITS;
const int8_t    FILTER_TABLE_FRACTION_BITS   = 30;
const int8_t    FILTER_TABLE_CUTOFF_EXT_BITS = 2;
const int8_t    FILTER_TABLE_RESO_EXT_BITS   = 2;
const uint8_t   EG_CONTROL_INTERVAL          = 0x10;
const int8_t    EG_LEVEL_MAX_BITS            = 30;
const int32_t   EG_LEVEL_MAX                 = 0x01 << EG_LEVEL_MAX_BITS;


const uint8_t   DATA_BYTE_MAX         = 0x7F;
const uint8_t   STATUS_BYTE_INVALID   = 0x7F;
const uint8_t   DATA_BYTE_INVALID     = 0x80;
const uint8_t   STATUS_BYTE_MIN       = 0x80;
const uint8_t   NOTE_OFF              = 0x80;
const uint8_t   NOTE_ON               = 0x90;
const uint8_t   CONTROL_CHANGE        = 0xB0;
const uint8_t   PROGRAM_CHANGE        = 0xC0;
const uint8_t   PITCH_BEND            = 0xE0;
const uint8_t   SYSTEM_MESSAGE_MIN    = 0xF0;
const uint8_t   SYSTEM_EXCLUSIVE      = 0xF0;
const uint8_t   TIME_CODE             = 0xF1;
const uint8_t   SONG_POSITION         = 0xF2;
const uint8_t   SONG_SELECT           = 0xF3;
const uint8_t   TUNE_REQUEST          = 0xF6;
const uint8_t   EOX                   = 0xF7;
const uint8_t   REAL_TIME_MESSAGE_MIN = 0xF8;
const uint8_t   ACTIVE_SENSING        = 0xFE;


const uint8_t   MODULATION      = 1;
const uint8_t   BTH_CONTROLLER  = 2;
const uint8_t   EXPRESSION      = 11;
const uint8_t   SUSTAIN_PEDAL   = 64;


const uint8_t   OSC_1_WAVE      = 102;
const uint8_t   MIXER_SUB_OSC   = 23;
const uint8_t   OSC_1_SHAPE     = 19;
const uint8_t   OSC_1_MORPH     = 20;

const uint8_t   OSC_2_WAVE      = 104;
const uint8_t   MIXER_OSC_MIX   = 21;
const uint8_t   OSC_2_COARSE    = 85;
const uint8_t   OSC_2_PITCH     = 76;

const uint8_t   FILTER_CUTOFF   = 74;
const uint8_t   FILTER_RESO     = 71;
const uint8_t   FILTER_EG_AMT   = 24;
const uint8_t   FILTER_KEY_TRK  = 9;

const uint8_t   EG_ATTACK       = 73;
const uint8_t   EG_DECAY        = 75;
const uint8_t   EG_SUSTAIN      = 30;
const uint8_t   EG_RELEASE      = 72;

const uint8_t   EG_OSC_AMT      = 89;
const uint8_t   EG_OSC_DST      = 8;
const uint8_t   VOICE_MODE      = 18;
const uint8_t   PORTAMENTO      = 5;

const uint8_t   LFO_WAVE        = 33;
const uint8_t   LFO_FADE_TIME   = 56;
const uint8_t   LFO_RATE        = 3;
const uint8_t   LFO_DEPTH       = 17;

const uint8_t   LFO_OSC_AMT     = 13;
const uint8_t   LFO_OSC_DST     = 103;
const uint8_t   LFO_FILTER_AMT  = 25;
const uint8_t   AMP_GAIN        = 15;

const uint8_t   AMP_ATTACK      = 52;
const uint8_t   AMP_DECAY       = 53;
const uint8_t   AMP_SUSTAIN     = 54;
const uint8_t   AMP_RELEASE     = 55;

const uint8_t   FILTER_MODE     = 39;
const uint8_t   P_BEND_RANGE    = 57;
const uint8_t   EG_AMP_MOD      = 36;
const uint8_t   REL_EQ_DECAY    = 105;

const uint8_t   BTH_FILTER_AMT  = 60;
const uint8_t   BTH_AMP_MOD     = 61;
const uint8_t   EG_VEL_SENS     = 62;
const uint8_t   AMP_VEL_SENS    = 63;

const uint8_t   AFT_T_LFO_AMT   = 109;
const uint8_t   VOICE_ASGN_MODE = 110;
const uint8_t   PAN             = 10;


const uint8_t   OSC_DRIFT       = 82;
const uint8_t   OSC_SAW_W_MODE  = 83;



const uint8_t   CHORUS_MIX      = 93;

const uint8_t   CHORUS_RATE     = 58;
const uint8_t   CHORUS_DEPTH    = 59;

const uint8_t   DELAY_LEVEL     = 94;
const uint8_t   DELAY_MODE      = 35;
const uint8_t   DELAY_TIME      = 90;
const uint8_t   DELAY_FEEDBACK  = 92;


const uint8_t   PROG_N_TO_W_TO  = 87;
const uint8_t   WRITE_P_TO_PROG = 106;
const uint8_t   PC_BY_CC_0      = 112;
const uint8_t   PC_BY_CC_1      = 113;
const uint8_t   PC_BY_CC_2      = 114;
const uint8_t   PC_BY_CC_3      = 115;
const uint8_t   PC_BY_CC_4      = 116;
const uint8_t   PC_BY_CC_5      = 117;
const uint8_t   PC_BY_CC_6      = 118;
const uint8_t   PC_BY_CC_7      = 119;


const uint8_t   ALL_SOUND_OFF   = 120;
const uint8_t   RESET_ALL_CTRLS = 121;
const uint8_t   ALL_NOTES_OFF   = 123;
const uint8_t   OMNI_MODE_OFF   = 124;
const uint8_t   OMNI_MODE_ON    = 125;
const uint8_t   MONO_MODE_ON    = 126;
const uint8_t   POLY_MODE_ON    = 127;

const uint8_t   OSC_WAVE_SAW       = 0;
const uint8_t   OSC_WAVE_SQUARE    = 25;
const uint8_t   OSC_WAVE_TRIANGLE  = 51;
const uint8_t   OSC_WAVE_SINE      = 76;
const uint8_t   OSC_WAVE_1_W_T     = 102;
const uint8_t   OSC_WAVE_2_O_1     = 102;
const uint8_t   OSC_WAVE_1_PULSE   = 127;
const uint8_t   OSC_WAVE_2_NOISE   = 127;

const uint8_t   MOD_DST_PITCH      = 0;
const uint8_t   MOD_DST_CUTOFF     = 25;
const uint8_t   MOD_DST_PITCH_2    = 76;
const uint8_t   MOD_DST_SHAPE_1    = 127;

const uint8_t   LFO_WAVE_TRIANGLE  = 0;
const uint8_t   LFO_WAVE_SINE      = 25;
const uint8_t   LFO_WAVE_RED_NOISE = 51;
const uint8_t   LFO_WAVE_SAW_DOWN  = 76;
const uint8_t   LFO_WAVE_S_AND_H   = 102;
const uint8_t   LFO_WAVE_SQUARE    = 127;

const uint8_t   VOICE_POLYPHONIC   = 0;
const uint8_t   VOICE_MONOPHONIC   = 76;
const uint8_t   VOICE_LEGATO       = 102;
const uint8_t   VOICE_LEGATO_PORTA = 127;


const uint8_t   PANEL_SCALE     = 128 + 8;
const uint8_t   PANEL_TRANSPOSE = 128 + 9;
const uint8_t   PANEL_PLAY_MODE = 128 + 10;
const uint8_t   PANEL_MIDI_CH   = 128 + 11;

const uint8_t   PANEL_PLAY_PIT  = 128 + 12;
const uint8_t   PANEL_PLAY_VELO = 128 + 13;
const uint8_t   PANEL_PIT_OFST  = 128 + 14;


const uint8_t   SEQ_PITCH_0     = 128 + 16;
const uint8_t   SEQ_PITCH_1     = 128 + 17;
const uint8_t   SEQ_PITCH_2     = 128 + 18;
const uint8_t   SEQ_PITCH_3     = 128 + 19;

const uint8_t   SEQ_PITCH_4     = 128 + 20;
const uint8_t   SEQ_PITCH_5     = 128 + 21;
const uint8_t   SEQ_PITCH_6     = 128 + 22;
const uint8_t   SEQ_PITCH_7     = 128 + 23;

const uint8_t   SEQ_PITCH_8     = 128 + 24;
const uint8_t   SEQ_PITCH_9     = 128 + 25;
const uint8_t   SEQ_PITCH_10    = 128 + 26;
const uint8_t   SEQ_PITCH_11    = 128 + 27;

const uint8_t   SEQ_PITCH_12    = 128 + 28;
const uint8_t   SEQ_PITCH_13    = 128 + 29;
const uint8_t   SEQ_PITCH_14    = 128 + 30;
const uint8_t   SEQ_PITCH_15    = 128 + 31;

const uint8_t   SEQ_VELO_0      = 128 + 32;
const uint8_t   SEQ_VELO_1      = 128 + 33;
const uint8_t   SEQ_VELO_2      = 128 + 34;
const uint8_t   SEQ_VELO_3      = 128 + 35;

const uint8_t   SEQ_VELO_4      = 128 + 36;
const uint8_t   SEQ_VELO_5      = 128 + 37;
const uint8_t   SEQ_VELO_6      = 128 + 38;
const uint8_t   SEQ_VELO_7      = 128 + 39;

const uint8_t   SEQ_VELO_8      = 128 + 40;
const uint8_t   SEQ_VELO_9      = 128 + 41;
const uint8_t   SEQ_VELO_10     = 128 + 42;
const uint8_t   SEQ_VELO_11     = 128 + 43;

const uint8_t   SEQ_VELO_12     = 128 + 44;
const uint8_t   SEQ_VELO_13     = 128 + 45;
const uint8_t   SEQ_VELO_14     = 128 + 46;
const uint8_t   SEQ_VELO_15     = 128 + 47;

const uint8_t   SEQ_TEMPO       = 128 + 48;
const uint8_t   SEQ_CLOCK_SRC   = 128 + 49;
const uint8_t   SEQ_GATE_TIME   = 128 + 50;
const uint8_t   SEQ_NUM_STEPS   = 128 + 51;

const uint8_t   SEQ_MODE        = 128 + 52;
const uint8_t   SEQ_ACT_STEPS   = 128 + 53;
const uint8_t   SEQ_PIT_OFST    = 128 + 54;
const uint8_t   SEQ_STEP_NOTE   = 128 + 55;

const uint8_t   SEQ_ON_STEPS    = 128 + 56;
const uint8_t   SEQ_TRX_ST_SP   = 128 + 57;



const uint8_t   RD_PROGRAM_0    = 128 + 64;
const uint8_t   RD_PROGRAM_1    = 128 + 65;
const uint8_t   RD_PROGRAM_2    = 128 + 66;
const uint8_t   RD_PROGRAM_3    = 128 + 67;

const uint8_t   RD_PROGRAM_4    = 128 + 68;
const uint8_t   RD_PROGRAM_5    = 128 + 69;
const uint8_t   RD_PROGRAM_6    = 128 + 70;
const uint8_t   RD_PROGRAM_7    = 128 + 71;

const uint8_t   RD_PROGRAM_8    = 128 + 72;
const uint8_t   RD_PROGRAM_9    = 128 + 73;
const uint8_t   RD_PROGRAM_10   = 128 + 74;
const uint8_t   RD_PROGRAM_11   = 128 + 75;

const uint8_t   RD_PROGRAM_12   = 128 + 76;
const uint8_t   RD_PROGRAM_13   = 128 + 77;
const uint8_t   RD_PROGRAM_14   = 128 + 78;
const uint8_t   RD_PROGRAM_15   = 128 + 79;

const uint8_t   WR_PROGRAM_0    = 128 + 80;
const uint8_t   WR_PROGRAM_1    = 128 + 81;
const uint8_t   WR_PROGRAM_2    = 128 + 82;
const uint8_t   WR_PROGRAM_3    = 128 + 83;

const uint8_t   WR_PROGRAM_4    = 128 + 84;
const uint8_t   WR_PROGRAM_5    = 128 + 85;
const uint8_t   WR_PROGRAM_6    = 128 + 86;
const uint8_t   WR_PROGRAM_7    = 128 + 87;

const uint8_t   WR_PROGRAM_8    = 128 + 88;
const uint8_t   WR_PROGRAM_9    = 128 + 89;
const uint8_t   WR_PROGRAM_10   = 128 + 90;
const uint8_t   WR_PROGRAM_11   = 128 + 91;

const uint8_t   WR_PROGRAM_12   = 128 + 92;
const uint8_t   WR_PROGRAM_13   = 128 + 93;
const uint8_t   WR_PROGRAM_14   = 128 + 94;
const uint8_t   WR_PROGRAM_15   = 128 + 95;

const uint8_t   RD_PANEL_PRMS   = 128 + 96;
const uint8_t   IN_PANEL_PRMS   = 128 + 97;

const uint8_t   WR_PANEL_PRMS   = 128 + 100;

const uint8_t   SEQ_RAND_PITCH  = 128 + 104;
const uint8_t   SEQ_RAND_VELO   = 128 + 105;
const uint8_t   PANIC_OP        = 128 + 106;
