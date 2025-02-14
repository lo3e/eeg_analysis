% eeg_event_segmenter.m
% ------------------------------------------------------------------------------
% Descrizione:
%   Divide il segnale EEG in due parti: la baseline (apertura e chiusura occhi) e
%   i segmenti EEG corrispondenti alle clip. Ogni clip viene estratta tra gli eventi
%   di inizio e fine clip, e viene gestita la rimozione di eventuali pause all'interno
%   delle clip. Gli eventi richiesti includono 'Eyes_Opened', 'Eyes_Closed',
%   'Clip_%_Start', 'Clip_%_End', e 'Start_Pausa'/'End_Pausa'.
%
% Funzioni:
%   eeg_event_segmenter  - Segmenta il segnale EEG in baseline e clip.
%   get_event_latency    - Restituisce la latenza di un evento specifico.
%   get_event_duration   - Restituisce la durata di un evento specifico.
%   remove_pauses        - Rimuove le pause all'interno di un segmento EEG.
%
% INPUT:
%   EEG (struct) : Struttura EEG (EEGLAB) contenente i dati e le informazioni sugli eventi.
%
% OUTPUT:
%   EEG_baseline (struct) : Segmento EEG relativo alla baseline (apertura e chiusura occhi).
%   EEG_Clips (cell array) : Cell array contenente i segmenti EEG corrispondenti alle clip.
%
% NOTE:
%   Eventi richiesti: 'Eyes_Opened', 'Eyes_Closed', 'Clip_%_Start', 'Clip_%_End'.
%   Gestisce anche eventuali pause tra gli eventi 'Start_Pausa' e 'End_Pausa'.
%
% Esempio:
%   [EEG_baseline, EEG_Clips] = eeg_event_segmenter(EEG)
%
% Autore: [Crescenzo Esposito]
% Data di creazione: [19/11/2024]
% ------------------------------------------------------------------------------
function [EEG_Clips,EEG_Levels] = eeg_event_segmenter(EEG,input_video,id,time_diff)

    % --- 1. Calcolo baseline ---
    try
        baseline_start = get_event_latency(EEG, 'Eyes_Opened') / EEG.srate;
        baseline_end = (get_event_latency(EEG, 'Eyes_Closed') + ...
                        get_event_duration(EEG, 'Eyes_Closed')) / EEG.srate;

        pop_rmbase(EEG,[baseline_start baseline_end]);
        %EEG_Baseline = pop_select(EEG, 'time', [baseline_start baseline_end]);
    catch ME
        warning('Errore nel calcolo della baseline: %s', ME.message);
    end

    % --- 2. Identificazione delle clip ---

    clip_start_idx = find([EEG.event.marker_value] >= 100 & [EEG.event.marker_value] < 200);
    num_clips = numel(clip_start_idx);

    if num_clips == 0
        warning('Nessuna clip trovata nei dati EEG.');
        EEG_Clips = {};
        return;
    end

    EEG_Clips = eeg_segmenter(EEG,num_clips,100,clip_start_idx,"clip",input_video,id,time_diff+EEG.event(1).init_time);
    % --- 3. Identificazione dei livelli ---
    level_start_idx = find([EEG.event.marker_value] >= 300 & [EEG.event.marker_value] < 400);
    num_levels = numel(level_start_idx);

    if num_levels == 0
        warning('Nessun livello trovato nei dati EEG.');
        EEG_Levels = {};
        return;
    end

    EEG_Levels = eeg_segmenter(EEG,num_levels,300,level_start_idx,"level",input_video,id,time_diff);

end

function latency = get_event_latency(EEG, event_type)
    idx = find(strcmp({EEG.event.type}, event_type), 1);
    if isempty(idx)
        error('Evento "%s" non trovato.', event_type);
    end
    latency = EEG.event(idx).latency;
end

function eeg_segments = eeg_segmenter(EEG,num_s,offset,s_start_idx,type,input_video,id,time_diff)
    eeg_segments = cell(1,num_s);
    s_count = 1;

    for i = 1:num_s
        try
            s_id = EEG.event(s_start_idx(i)).marker_value - offset;
            end_idx = find([EEG.event.marker_value] == offset + 100 + s_id & ...
                           [EEG.event.latency] > EEG.event(s_start_idx(i)).latency, 1);

            if isempty(end_idx)
                warning('Clip %d non ha un evento di fine. Ignorata.', s_id);
                continue;
            end

            s_start = EEG.event(s_start_idx(i)).init_time;
            s_end = EEG.event(end_idx).init_time;

            EEG_segment = pop_select(EEG, 'time', [s_start s_end]);
            if (type == "clip")
                EEG_segment.setname = ['Clip_',num2str(s_id)];
                % Rimuovi pause all'interno della clip
                %EEG_segment = remove_pauses(EEG_segment, EEG, s_start, s_end);
                output_video_dir = fullfile('Video_Segments',id,'video'); % Modifica con la tua directory
                video_cutter(EEG_segment.setname,input_video,s_start+time_diff,s_end+time_diff,i,output_video_dir);

            else
                EEG_segment.setname = [EEG.event(end_idx).type];
                output_video_dir = fullfile('Video_Segments',id,'Levels'); % Modifica con la tua directory
                video_cutter(EEG_segment.setname,input_video,s_start+time_diff,s_end+time_diff,i,output_video_dir);
            end
            
            
            EEG_segment.event = [];
            eeg_segments{s_count} = EEG_segment;
            s_count = s_count + 1;

        catch ME
            warning('Errore nell''elaborazione del segmento %d: %s', s_id, ME.message);
        end
    end
end

function video_cutter(name,input_video,s_start,s_end,i,output_video_dir)

    % Percorso di output per il video segmentato
    if ~exist(output_video_dir, 'dir')
        mkdir(output_video_dir);
    end
    duration = s_end - s_start;
    % Estrai il segmento video corrispondente

    video_output_path = fullfile(output_video_dir, sprintf('%d_%s.mp4',i,name));
    ffmpeg_command = sprintf('ffmpeg -n -ss %.3f -accurate_seek -i "%s" -t %.3f -c:v copy -c:a copy -movflags +faststart -fflags +genpts "%s" ',s_start, input_video, duration, video_output_path);

    %fprintf('Esecuzione comando FFmpeg:\n%s\n', ffmpeg_command);            
    % Esegui FFmpeg
    [status, cmdout] = system(ffmpeg_command);
    %disp(cmdout)
    if status ~= 0
        warning('Errore durante l''estrazione del video per %s: %s', name, cmdout);
    else
        fprintf('Video per %s salvato in: %s\n', name, video_output_path);
    end
            
end

function duration = get_event_duration(EEG, event_type)
    idx = find(strcmp({EEG.event.type}, event_type), 1);
    if isempty(idx) || ~isfield(EEG.event(idx), 'duration')
        error('Durata non trovata per l''evento "%s".', event_type);
    end
    duration = EEG.event(idx).duration;
end

function EEG_segment = remove_pauses(EEG_segment, EEG, clip_start, clip_end)
    pauses = find([EEG.event.marker_value] == 10 & ...
                  [EEG.event.latency] > clip_start * EEG.srate & ...
                  [EEG.event.latency] < clip_end * EEG.srate);

    for p = 1:length(pauses)
        pause_start = (EEG.event(pauses(p)).latency / EEG.srate) - clip_start;
        pause_end_idx = find( [EEG.event.marker_value] == 11 & ...
                             [EEG.event.latency] > EEG.event(pauses(p)).latency, 1);

        if isempty(pause_end_idx)
            warning('Fine pausa non trovata. Ignorata.');
            continue;
        end

        pause_end = (EEG.event(pause_end_idx).latency / EEG.srate) - clip_start;
        EEG_segment = pop_select(EEG_segment, 'rmtime', [pause_start - 1 pause_end + 1]);
    end
end