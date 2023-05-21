#!/bin/bash

while getopts :d:i:t:m:x:o:n:s arg; do
    case "$arg" in
        (d) 
            dataset_path=${OPTARG}
            ;;
        (i)
            folder_in=${OPTARG}
            ;;
        (t) 
            target_fs=${OPTARG}
            ;;
        (m)
            ind_min=${OPTARG}
            ;;
        (x)
            ind_max=${OPTARG}
            ;;
        (o)  
            output_folder=${OPTARG}
            ;;
        (n) 
            new_audioFileDuration=${OPTARG}
            ;;
        (s) 
            pad_silence=1
            ;;
        (h) 
            usage
            exit 0
            ;;
        (*)  
            usage
            exit 1
            ;;
    esac
done

[[ -v pad_silence ]] && pad_silence=0 # If -s is not provided, then we don't pad the silence

cd "$dataset_path/raw/audio/"

SOX_PATH="/appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox"
## NEW
FILE="$dataset_path/analysis/subset_files.csv"

if test -f "$FILE"; then
    while IFS=';' read -ra array; do
      ar1+=("${array[0]}")
    done < $FILE  
    listwav=$(printf $folder_in'/%s\n' "${ar1[@]}")
else
    listwav=$(ls $folder_in/*.wav)
fi
##


COUNTER=0
for f in $listwav;

    do

        echo "current wavfile: "$f

        COUNTSEG=0

        FILENAME=${f##*/}
        FILENAME=${FILENAME%.*}
                
        orig_samp=$($SOX_PATH --i -s "${f}")
        orig_fs=$($SOX_PATH --i -r "${f}")
        orig_dura=$(echo "scale=4; $orig_samp / $orig_fs" | bc)
        orig_dura_in_s=$(($orig_samp / $orig_fs))

        echo "original duration: "$orig_dura


        COUNTER=$[$COUNTER +1]
        
        if [ "$COUNTER" -ge "$ind_min" ] && [ "$COUNTER" -le "$ind_max" ];# && [ ! -f $3"/${FILENAME}" ]  
        
        then         

            # CASE 1 : no segmentation done, as new audio file duration is equal to the original one (comparison done in seconds)
            if [ $new_audioFileDuration -eq $orig_dura_in_s ]; then 
            
                $SOX_PATH "${f}" -r $target_fs -t wavpcm $folder_out"/${FILENAME}.wav"
                
                echo $folder_out"/${FILENAME}.wav"

            # CASE 2 : perform segmentation
            else                

                for indi in `seq 0 $new_audioFileDuration $orig_dura`;       

                do
                    printf -v count "%03d" "$COUNTSEG" # pad $COUNTSEG with leading 0
                    nn="_seg$count"

                    bo2=$(echo "scale=4;$(($indi+$new_audioFileDuration))<$orig_dura" | bc)

                    # SEGMENTATION CASE 1 : all segments inside the original wav 
                    if [ $bo2 = 1 ]; then
    
                        $SOX_PATH "${f}" -r $target_fs -t wavpcm $output_folder"/${FILENAME}$nn.wav" trim $indi $new_audioFileDuration      

                        echo ">> /${FILENAME}$nn.wav"
                        echo ">> trim from" $indi "to" $(($indi+$new_audioFileDuration))        

                    # SEGMENTATION CASE 2 : for the last segment which will end after the original wav
                    else

                        echo ">> /${FILENAME}$nn.wav"

                        # CASE with silence padding, keeping a same segment duration
                        if [ $pad_silence = 1 ]; then

                            dura_silence=$(echo "scale=4;$(($indi+$new_audioFileDuration)) - $orig_dura" | bc)

                            $SOX_PATH "${f}" -r $target_fs -t wavpcm $output_folder"/${FILENAME}$nn.wav" trim $indi $new_audioFileDuration pad 0 $dura_silence  

                            echo ">> trim from" $indi "to" $(($indi+$new_audioFileDuration))   
                            echo ">> add silence of" $dura_silence "s"

                        # CASE without silence padding, reducing the segment duration
                        else

                            $SOX_PATH "${f}" -r $target_fs -t wavpcm $output_folder"/${FILENAME}$nn.wav" trim $indi $new_audioFileDuration 

                            # echo ">> trim from" $indi "to" $(($indi+$new_audioFileDuration-$dura_silence) | bc)    

                        fi

       
                    fi
    
                COUNTSEG=$[$COUNTSEG +1]
    
                done # end of segmentation
                
            fi
        fi
                
    done