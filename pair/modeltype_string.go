// Code generated by "stringer -type=ModelType"; DO NOT EDIT.

package pair

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[AttackerType-0]
	_ = x[JudgeType-1]
	_ = x[TargetType-2]
}

const _ModelType_name = "AttackerTypeJudgeTypeTargetType"

var _ModelType_index = [...]uint8{0, 12, 21, 31}

func (i ModelType) String() string {
	if i < 0 || i >= ModelType(len(_ModelType_index)-1) {
		return "ModelType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _ModelType_name[_ModelType_index[i]:_ModelType_index[i+1]]
}
